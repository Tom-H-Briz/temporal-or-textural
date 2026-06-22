"""
Feature visualiser: overlay SAE feature activations on video frames.

For each (feature_id, clip_id) pair, runs a forward pass, extracts per-token
activations for the feature, maps tokens to 16×16 spatiotemporal patches, and
saves a 4×4 PNG grid of overlaid frames.

Outputs:
  outputs/analysis/visualiser/feat{feature_id}_clip{clip_id}.png
"""

import os
import sys
from pathlib import Path

import av
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.dfa_engine import DFAEngine

CFG = {
    "model_flag": "videomae",
    "sae_path": str(ROOT / "outputs/sae/sae_layer7_job128.pt"),
    "dim_mean_path": str(ROOT / "outputs/sae/layer7_dim_mean.pt"),
    "layer": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "output_dir": str(ROOT / "outputs/analysis/visualiser/"),
    "pairs": [
        # (feature_id, clip_id)
        (4744, "22922"),
    ],
    "positive_only": True,
    "overlay_alpha": 0.5,
    "colormap": "plasma",
}

NUM_FRAMES = 16
PATCH_SIZE = 16        # spatial patch size in pixels
T_PATCHES  = 8        # temporal patches (16 frames / tubelet_size 2)
S_PATCHES  = 14       # spatial patches per axis (224 / 16)
FRAME_SIZE = 224


def load_frames(clip_path: Path) -> list[np.ndarray]:
    """Decode all frames and sample 16 evenly, resized to 224×224, uint8 RGB."""
    container = av.open(str(clip_path))
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n = len(all_frames)
    indices = torch.linspace(0, n - 1, NUM_FRAMES).long().tolist()
    frames = []
    for i in indices:
        img = Image.fromarray(all_frames[i]).resize((FRAME_SIZE, FRAME_SIZE), Image.BILINEAR)
        frames.append(np.array(img))
    return frames  # list of 16 × (224, 224, 3) uint8


def build_activation_map(z: torch.Tensor, feature_id: int, positive_only: bool) -> np.ndarray:
    """
    z: (1568, dict_size)
    Returns activation_map (16, 224, 224) float32, normalised to [0, 1].
    Token ordering: token = t * S_PATCHES^2 + y * S_PATCHES + x
    """
    activations = z[:, feature_id].cpu().float().numpy()   # (1568,)
    if positive_only:
        activations = np.clip(activations, 0, None)

    act_map = np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE), dtype=np.float32)
    for token_idx in range(len(activations)):
        t = token_idx // (S_PATCHES * S_PATCHES)
        remainder = token_idx % (S_PATCHES * S_PATCHES)
        y = remainder // S_PATCHES
        x = remainder % S_PATCHES
        val = activations[token_idx]
        y0, y1 = y * PATCH_SIZE, (y + 1) * PATCH_SIZE
        x0, x1 = x * PATCH_SIZE, (x + 1) * PATCH_SIZE
        # tubelet_size=2: t maps to frames t*2 and t*2+1
        act_map[t * 2,     y0:y1, x0:x1] = val
        act_map[t * 2 + 1, y0:y1, x0:x1] = val

    act_max = act_map.max()
    if act_max > 0:
        act_map /= act_max
    return act_map


def overlay_frame(frame: np.ndarray, act: np.ndarray, alpha: float, cmap: str) -> np.ndarray:
    """Blend RGB frame with heatmap. Returns (224, 224, 3) uint8."""
    colormap = plt.get_cmap(cmap)
    heat = (colormap(act)[:, :, :3] * 255).astype(np.uint8)
    return (frame * (1 - alpha) + heat * alpha).astype(np.uint8)


def save_grid(frames: list[np.ndarray], act_map: np.ndarray, feature_id: int, clip_id: str, cfg: dict, out_dir: Path) -> None:
    alpha = cfg["overlay_alpha"]
    cmap  = cfg["colormap"]
    overlaid = [overlay_frame(frames[i], act_map[i], alpha, cmap) for i in range(NUM_FRAMES)]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(overlaid[i])
        ax.set_title(f"f{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Feature {feature_id} — Clip {clip_id}", fontsize=12)
    fig.tight_layout()

    out_path = out_dir / f"feat{feature_id}_clip{clip_id}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def main() -> None:
    cfg = CFG
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(cfg["video_dir"])

    with DFAEngine(
        model_flag=cfg["model_flag"],
        sae_path=cfg["sae_path"],
        dim_mean_path=cfg["dim_mean_path"],
        layer=cfg["layer"],
        device=cfg["device"],
    ) as engine:
        for feature_id, clip_id in cfg["pairs"]:
            clip_path = video_dir / f"{clip_id}.webm"
            frames   = load_frames(clip_path)
            z        = engine.get_z(clip_path)
            act_map  = build_activation_map(z, feature_id, cfg["positive_only"])
            save_grid(frames, act_map, feature_id, clip_id, cfg, out_dir)


if __name__ == "__main__":
    main()
