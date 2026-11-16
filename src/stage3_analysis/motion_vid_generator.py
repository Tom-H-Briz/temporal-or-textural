import json
import math
import av
from pathlib import Path
import numpy as np
import torch

CFG = {
    "out_dir": "outputs/synthetic_motion/directional_clips",
    "size": 224,
    "fps": 24,
    "duration_sec": 5.0,
    "umt_frames": 12,
    "bg_val": 0.175,
    "shapes": ["ball", "pyramid"],
    # Screen coords, y points DOWN: 0 = right, 90 = DOWN, 180 = left, 270 = UP (degrees)
    "directions_deg": [0, 45, 90, 135, 180, 225, 270, 315],
    # Total travel across the 12 sampled frames: slow = 4 px/step, fast = 10 px/step
    "total_travel_px": {"slow_center": 44.0, "fast_center": 110.0},
    "imagenet_mean": (0.485, 0.456, 0.406),
    "imagenet_std": (0.229, 0.224, 0.225),
}
# Two-object suite (16/11/26, #5216 test): one object static at the centre, the other moves horizontally past it,
# either behind it (same row, occluded) or side by side (own lane below); controls = mover alone, both static
PAIR_CFG = dict(CFG, out_dir="outputs/synthetic_motion/pair_clips",
                pair_travel_px=150.0,    # mover starts / ends clear of the static object (half-widths ~41 px)
                lane_offset_px=56.0)     # side-by-side lane: no overlap with the static object
# Same two-object layouts with coloured objects (16/11/26): red ball, blue pyramid (RGB multipliers on the shading)
COLOR_PAIR_CFG = dict(PAIR_CFG, out_dir="outputs/synthetic_motion/pair_clips_color",
                      colors={"ball": (1.0, 0.15, 0.15), "pyramid": (0.2, 0.35, 1.0)})


def _point_in_triangle_sdf(px: np.ndarray, py: np.ndarray, v0, v1, v2) -> np.ndarray:
    """Exact signed distance (negative inside, positive outside) to a 2D triangle."""
    verts = [np.array(v0, dtype=np.float32), np.array(v1, dtype=np.float32), np.array(v2, dtype=np.float32)]
    d_min = np.full_like(px, np.inf)

    for i in range(3):
        a = verts[i]
        b = verts[(i + 1) % 3]
        e = b - a
        wx = px - a[0]
        wy = py - a[1]

        t = np.clip((wx * e[0] + wy * e[1]) / (e[0] * e[0] + e[1] * e[1]), 0.0, 1.0)
        dx = wx - e[0] * t
        dy = wy - e[1] * t
        d_min = np.minimum(d_min, dx * dx + dy * dy)

    inside = _inside_triangle(px, py, v0, v1, v2)
    return np.where(inside, -np.sqrt(d_min), np.sqrt(d_min))


def _inside_triangle(px: np.ndarray, py: np.ndarray, v0, v1, v2) -> np.ndarray:
    """True where the point is on the same side of all 3 edges (works for either vertex winding)."""
    verts = [v0, v1, v2]
    crosses = []
    for i in range(3):
        a, b = verts[i], verts[(i + 1) % 3]
        crosses.append((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]))
    all_pos = (crosses[0] > 0) & (crosses[1] > 0) & (crosses[2] > 0)
    all_neg = (crosses[0] < 0) & (crosses[1] < 0) & (crosses[2] < 0)
    return all_pos | all_neg


def render_object_frame(
    shape: str,
    H: int,
    W: int,
    cx: float,
    cy: float,
    vx: float,
    vy: float,
    bg_val: float | np.ndarray = 0.175,
    shutter_samples: int = 8,
    shutter_fraction: float = 0.5,
    edge_softness: float = 1.5,
    tint: tuple = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Renders either a 3D-shaded sphere ('ball') or faceted tetrahedron ('pyramid') -> (H, W, 3).
    bg_val: scalar grey or an (H, W, 3) image (for layering); tint: RGB multiplier on the object's shading."""
    light_dir = np.array([-0.45, -0.55, 0.70], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    half_vec = light_dir + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec /= np.linalg.norm(half_vec)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    accum = np.zeros((H, W, 3), dtype=np.float32)
    tint_rgb = np.array(tint, dtype=np.float32)
    offsets = np.linspace(-0.5 * shutter_fraction, 0.5 * shutter_fraction, shutter_samples)

    ball_r = 16.0
    # Equilateral triangle area = (3*sqrt(3)/4) * R^2; solve R so it equals the ball's pi * r^2 (~804 px^2)
    pyr_r = ball_r * math.sqrt(math.pi / (3.0 * math.sqrt(3.0) / 4.0))  # ~24.9

    for dt in offsets:
        x_pos = cx + vx * dt
        y_pos = cy + vy * dt

        if shape == "ball":
            dx = xx - x_pos
            dy = yy - y_pos
            dist = np.sqrt(dx * dx + dy * dy)
            alpha = np.clip((ball_r + edge_softness * 0.5 - dist) / edge_softness, 0.0, 1.0)

            nx = np.clip(dx / ball_r, -1.0, 1.0)
            ny = np.clip(dy / ball_r, -1.0, 1.0)
            nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny))

            diffuse = np.maximum(0.0, nx * light_dir[0] + ny * light_dir[1] + nz * light_dir[2])
            spec = np.maximum(0.0, nx * half_vec[0] + ny * half_vec[1] + nz * half_vec[2]) ** 24
            obj_intensity = np.clip(0.25 + 0.58 * diffuse + 0.14 * spec, 0.0, 1.0)

        elif shape == "pyramid":
            # Upright triangle in screen coordinates (y points down -> CCW order: top, left, right)
            v_top = (x_pos, y_pos - pyr_r)
            v_left = (x_pos - pyr_r * math.cos(math.radians(30)), y_pos + pyr_r * math.sin(math.radians(30)))
            v_right = (x_pos + pyr_r * math.cos(math.radians(30)), y_pos + pyr_r * math.sin(math.radians(30)))
            apex = (x_pos, y_pos - 0.12 * pyr_r)

            sdf = _point_in_triangle_sdf(xx, yy, v_top, v_left, v_right)
            alpha = np.clip((edge_softness * 0.5 - sdf) / edge_softness, 0.0, 1.0)

            # Each facet is the sub-triangle (apex, vertex, next vertex)
            in_left = _inside_triangle(xx, yy, apex, v_top, v_left)
            in_bot = _inside_triangle(xx, yy, apex, v_left, v_right)

            # Facet normals (left-lit, bottom-sloped, right-shadowed)
            n_left = np.array([-0.62, -0.35, 0.70], dtype=np.float32)
            n_right = np.array([0.62, -0.35, 0.70], dtype=np.float32)
            n_bot = np.array([0.0, 0.72, 0.69], dtype=np.float32)

            i_left = np.clip(0.25 + 0.65 * max(0.0, float(np.dot(n_left, light_dir))), 0.0, 1.0)
            i_right = np.clip(0.25 + 0.65 * max(0.0, float(np.dot(n_right, light_dir))), 0.0, 1.0)
            i_bot = np.clip(0.25 + 0.65 * max(0.0, float(np.dot(n_bot, light_dir))), 0.0, 1.0)

            # Assign facet intensity cleanly
            obj_intensity = np.where(
                in_left,
                i_left,
                np.where(in_bot, i_bot, i_right),
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")

        a = alpha[..., None]
        accum += bg_val * (1.0 - a) + (obj_intensity * alpha)[..., None] * tint_rgb

    return accum / shutter_samples


def render_clip(shape: str, deg, travel: float, idx: np.ndarray, cfg: dict):
    """Renders the full-rate clip as uint8 (T, H, W, 3). deg=None -> stationary at the centre."""
    n_full, size = int(cfg["fps"] * cfg["duration_sec"]), cfg["size"]
    ux, uy = (0.0, 0.0) if deg is None else (math.cos(math.radians(deg)), math.sin(math.radians(deg)))
    v_full = travel / float(idx[-1] - idx[0])  # px per full-rate frame
    v_step = travel / (len(idx) - 1)  # px per sampled step (sets the shutter blur)
    t_mid, centre = (n_full - 1) / 2.0, size / 2.0
    frames, coords = [], []
    for t in range(n_full):
        cx, cy = centre + (t - t_mid) * ux * v_full, centre + (t - t_mid) * uy * v_full
        img = render_object_frame(shape, size, size, cx, cy, ux * v_step, uy * v_step, cfg["bg_val"])
        frames.append(np.round(img * 255.0).astype(np.uint8))
        coords.append([round(cx, 3), round(cy, 3)])
    return np.stack(frames), coords, v_step


def to_model_tensor(frames_u8: np.ndarray, idx: np.ndarray, cfg: dict) -> torch.Tensor:
    """Sampled frames -> ImageNet-normalised (T, C, H, W), the layout the UMT wrapper expects."""
    x = torch.from_numpy(frames_u8[idx]).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(cfg["imagenet_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["imagenet_std"]).view(1, 3, 1, 1)
    return (x - mean) / std


def write_webm(frames_u8: np.ndarray, path: Path, fps: int):
    """Writes uint8 (T, H, W, 3) frames as a lossless VP9 webm (same container as SSv2)."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libvpx-vp9", rate=fps)
        stream.width, stream.height = frames_u8.shape[2], frames_u8.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"lossless": "1"}
        for frame in frames_u8:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())


def make_clip(shape: str, deg, label: str, travel: float, idx: np.ndarray, cfg: dict, out: Path):
    """Renders one clip, saves compressed full-rate frames + the model tensor, returns (tensor, metadata)."""
    frames_u8, coords, v_step = render_clip(shape, deg, travel, idx, cfg)
    model_x = to_model_tensor(frames_u8, idx, cfg)
    name = f"{shape}_static_center" if deg is None else f"{shape}_dir_{deg:03d}_{label}"
    write_webm(frames_u8, out / "full_rate" / f"{name}.webm", cfg["fps"])
    torch.save(model_x, out / "umt_12f" / f"{name}.pt")
    meta = {
        "name": name, "shape": shape, "angle_deg": deg, "speed_label": label,
        "umt_step_px": round(v_step, 3), "fps": cfg["fps"], "num_full_frames": len(frames_u8),
        "umt_sampled_indices": idx.tolist(), "umt_trajectory_xy": [coords[i] for i in idx],
    }
    return model_x, meta


def generate_suite(cfg: dict = CFG):
    """Moving clips (shape x direction x speed) plus one stationary centred clip per shape."""
    out = Path(cfg["out_dir"])
    (out / "full_rate").mkdir(parents=True, exist_ok=True)
    (out / "umt_12f").mkdir(parents=True, exist_ok=True)
    n_full = int(cfg["fps"] * cfg["duration_sec"])
    idx = np.linspace(0, n_full - 1, cfg["umt_frames"]).round().astype(int)
    jobs = [(s, d, lab, tr) for s in cfg["shapes"] for d in cfg["directions_deg"]
            for lab, tr in cfg["total_travel_px"].items()]
    jobs += [(s, None, "static", 0.0) for s in cfg["shapes"]]
    tensors, metadata = [], []
    for shape, deg, label, travel in jobs:
        x, meta = make_clip(shape, deg, label, travel, idx, cfg, out)
        tensors.append(x)
        metadata.append({"clip_id": len(metadata), **meta})
    batch = torch.stack(tensors)  # (N, T, C, H, W)
    torch.save(batch, out / "all_umt_clips.pt")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {tuple(batch.shape)} -> {out / 'all_umt_clips.pt'}")
    return batch, metadata


def render_scene(layers, idx: np.ndarray, cfg: dict):
    """layers back-to-front: [(shape, (x_mid, y_mid), (ux, uy), travel)] -> uint8 (T, H, W, 3) + per-layer xy.
    Each layer is drawn over the image so far (render_object_frame's bg_val broadcasts), so later layers occlude."""
    n_full, size = int(cfg["fps"] * cfg["duration_sec"]), cfg["size"]
    t_mid, span, steps = (n_full - 1) / 2.0, float(idx[-1] - idx[0]), len(idx) - 1
    frames, coords = [], []
    for t in range(n_full):
        img, pos = np.full((size, size, 3), cfg["bg_val"], np.float32), []
        for shape, (x0, y0), (ux, uy), travel in layers:
            cx, cy = x0 + (t - t_mid) * ux * travel / span, y0 + (t - t_mid) * uy * travel / span
            v_step = travel / steps
            tint = cfg.get("colors", {}).get(shape, (1.0, 1.0, 1.0))      # grey unless the cfg colours it
            img = render_object_frame(shape, size, size, cx, cy, ux * v_step, uy * v_step, img, tint=tint)
            pos.append([round(cx, 2), round(cy, 2)])
        frames.append(np.round(img * 255.0).astype(np.uint8))
        coords.append(pos)
    return np.stack(frames), coords


def pair_jobs(cfg: dict):
    """[(name, layers)]: per (mover, static) role and direction (0 = LtR, 180 = RtL): behind, side, mover alone
    on each path; plus both static side by side per role."""
    c, lane, T = cfg["size"] / 2.0, cfg["size"] / 2.0 + cfg["lane_offset_px"], cfg["pair_travel_px"]
    jobs = []
    for mover, still in (("ball", "pyramid"), ("pyramid", "ball")):
        for deg, u in ((0, (1.0, 0.0)), (180, (-1.0, 0.0))):
            jobs += [(f"{mover}_behind_{deg:03d}", [(mover, (c, c), u, T), (still, (c, c), (0, 0), 0.0)]),
                     (f"{mover}_side_{deg:03d}", [(still, (c, c), (0, 0), 0.0), (mover, (c, lane), u, T)]),
                     (f"{mover}_alone_center_{deg:03d}", [(mover, (c, c), u, T)]),
                     (f"{mover}_alone_lane_{deg:03d}", [(mover, (c, lane), u, T)])]
        jobs.append((f"{mover}_static_side", [(still, (c, c), (0, 0), 0.0), (mover, (c, lane), (0, 0), 0.0)]))
    return jobs


def generate_pair_suite(cfg: dict = PAIR_CFG):
    """Two-object clips: webm (full rate) + model tensors + stacked batch + metadata, as generate_suite."""
    out = Path(cfg["out_dir"])
    (out / "full_rate").mkdir(parents=True, exist_ok=True)
    (out / "umt_12f").mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, int(cfg["fps"] * cfg["duration_sec"]) - 1, cfg["umt_frames"]).round().astype(int)
    tensors, metadata = [], []
    for name, layers in pair_jobs(cfg):
        frames_u8, coords = render_scene(layers, idx, cfg)
        x = to_model_tensor(frames_u8, idx, cfg)
        write_webm(frames_u8, out / "full_rate" / f"{name}.webm", cfg["fps"])
        torch.save(x, out / "umt_12f" / f"{name}.pt")
        tensors.append(x)
        metadata.append({"clip_id": len(metadata), "name": name, "layers": [list(map(str, l[:1])) + list(l[1:]) for l in layers],
                         "umt_sampled_indices": idx.tolist(), "umt_trajectory_xy": [coords[i] for i in idx]})
    torch.save(torch.stack(tensors), out / "all_umt_clips.pt")
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(tensors)} pair clips -> {out}")


if __name__ == "__main__":
    import sys
    if "pair_color" in sys.argv:
        generate_pair_suite(COLOR_PAIR_CFG)
    elif "pair" in sys.argv:
        generate_pair_suite()
    else:
        generate_suite()
