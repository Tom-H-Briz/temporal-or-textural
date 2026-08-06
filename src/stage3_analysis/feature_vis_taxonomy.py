"""
Taxonomy example clips — cross-backbone visual check (CC brief 06/08, Phase 2).

Batch-generates one Clip/Real/Shuffle/Still activation figure per (feature, clip)
candidate from the Phase 1 selection, both VM and TF. Not built on feature_vis_vm.py
— TF differs enough (8 frames not 16, no tubelet-pairing, full-permutation shuffle
not paired-shuffle, cls_offset=1 not 0) that forcing shared code was worse than a
plain from-scratch script (Tom, 06/08: "not so fussy about code reuse on this").

Usage:
    uv run python src/stage3_analysis/feature_vis_taxonomy.py
"""

import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))

from sae import BatchTopKSAE
from perturbation import apply_shuffle
from ToT_utils import (
    CHECKPOINT_REGISTRY, FRAME_SAMPLERS, MODEL_REGISTRY, _deterministic_seed,
    gather_by_position, resolve_sae_checkpoint,
)

OUT_DIR = ROOT / "outputs" / "analysis" / "feature_vis_taxonomy"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
INCLUDE_VALUES_IN_CAPTION = True  # brief's default, pending explicit override

BACKBONE_CFG = {
    "vm": dict(
        model_flag="videomae", num_frames=16,
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_l7_job7ep_k64.parquet",
        video_dir=ROOT / "data/ssv2/20bn-something-something-v2",
        shuffle_label="C1",
    ),
    "tf": dict(
        model_flag="timesformer", num_frames=8,
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
        video_dir=ROOT / "data/ssv2/20bn-something-something-v2",
        shuffle_label="C",
    ),
}

# Transcribed directly from the confirmed Phase 1 candidate table (06/08).
# (backbone, class_id, clip_id, feature_idx, bucket, signed_R, signed_shuffle)
CANDIDATES = [
    ("vm", 123, "138724", 1265, "noise",     0.2067, 0.1966),
    ("vm", 123, "138724", 1989, "sign_flip", 0.1450, -0.1548),  # swapped from 4738 (06/08): failed purity guard
    ("vm", 123, "138724", 805,  "decrease",  -0.2565, -0.0108),
    ("vm", 123, "138724", 4256, "increase",  0.1886, 0.4259),
    ("vm", 126, "198950", 377,  "noise",     0.0468, 0.0474),
    ("vm", 126, "198950", 3347, "sign_flip", -0.0590, 0.0231),
    ("vm", 126, "198950", 5490, "decrease",  -0.0852, -0.0472),
    ("vm", 126, "198950", 2197, "increase",  0.0751, 0.0984),
    ("vm", 6,   "162359", 0,    "noise",     0.1148, 0.1150),
    ("vm", 6,   "162359", 4233, "sign_flip", -0.1228, 0.1838),
    ("vm", 6,   "162359", 6141, "decrease",  0.1503, 0.0313),
    ("vm", 6,   "162359", 354,  "increase",  0.1481, 0.2391),
    ("vm", 0,   "50492",  3336, "noise",     -0.1480, -0.1411),
    ("vm", 0,   "50492",  1989, "sign_flip", 0.0996, -0.2050),
    ("vm", 0,   "50492",  3608, "decrease",  -0.2821, -0.2083),
    ("vm", 0,   "50492",  1219, "increase",  0.1212, 0.2868),
    # swapped from clip 208136 (06/08): correct data, but near-static clip content
    ("vm", 59,  "120137", 6067, "noise",     0.1247, -0.1303),
    ("vm", 59,  "120137", 1265, "sign_flip", -0.3143, 1.6024),
    ("vm", 59,  "120137", 377,  "decrease",  0.1447, 0.1028),
    ("vm", 59,  "120137", 6090, "increase",  -0.1457, -0.3175),

    ("tf", 41,  "139808", 2156, "noise",     -1.6727, -1.7106),
    ("tf", 41,  "139808", 2090, "sign_flip", 1.5109, -1.2019),
    ("tf", 41,  "139808", 2057, "decrease",  3.5428, 2.4982),
    ("tf", 41,  "139808", 1588, "increase",  1.7915, 2.4680),
    ("tf", 126, "84491",  1588, "noise",     -1.8041, -1.7285),
    # no sign_flip candidate exists in this clip/class — confirmed structural, not a search gap
    ("tf", 126, "84491",  3029, "decrease",  1.8277, 1.6900),
    ("tf", 126, "84491",  2156, "increase",  -1.8353, -2.1188),
    ("tf", 32,  "167318", 622,  "noise",     3.4521, 3.4233),
    ("tf", 32,  "167318", 6029, "sign_flip", -1.2397, 0.4491),
    ("tf", 32,  "167318", 4590, "decrease",  4.4326, 4.1141),
    ("tf", 32,  "167318", 2057, "increase",  -2.3151, -2.4889),
    ("tf", 6,   "218395", 1746, "noise",     -1.3077, -1.3668),
    ("tf", 6,   "218395", 2156, "sign_flip", -1.6721, 0.5490),
    ("tf", 6,   "218395", 3813, "decrease",  1.2985, 0.9591),
    ("tf", 6,   "218395", 1588, "increase",  -1.9997, -2.4365),
    ("tf", 29,  "20535",  622,  "noise",     1.9209, 1.9050),
    ("tf", 29,  "20535",  1517, "sign_flip", -1.4800, 0.0355),
    ("tf", 29,  "20535",  3813, "decrease",  1.2621, 1.0301),
    ("tf", 29,  "20535",  1588, "increase",  -2.0946, -3.1523),
]


def load_model_and_sae(backbone_cfg: dict, layer: int = 7):
    model_flag = backbone_cfg["model_flag"]
    resolved = resolve_sae_checkpoint(model_flag, layer, dataset_name="ssv2", sae_k=64)
    model_cfg = MODEL_REGISTRY[model_flag]
    checkpoint = CHECKPOINT_REGISTRY[(model_flag, "ssv2")]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(DEVICE).eval().requires_grad_(False)

    ckpt = torch.load(resolved["sae_path"], weights_only=True, map_location=DEVICE)
    state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    top_k = resolved["sae_k"] * model_cfg["num_patch_tokens"]
    sae = BatchTopKSAE(input_shape=model_cfg["hidden_dim"], nb_concepts=nb_concepts,
                       top_k=top_k, device=DEVICE)
    sae.load_state_dict(state_dict)
    dim_mean = torch.load(resolved["dim_mean_path"], weights_only=True, map_location=DEVICE)
    sae.train()
    dummy = torch.zeros(model_cfg["num_patch_tokens"], model_cfg["hidden_dim"], device=DEVICE)
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval().requires_grad_(False)
    return model, processor, sae, dim_mean


def load_raw_frames(clip_id: str, video_dir: Path) -> list:
    container = av.open(str(video_dir / f"{clip_id}.webm"))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def sample_frames(raw_frames: list, num_frames: int) -> list:
    idx = FRAME_SAMPLERS["ssv2"](len(raw_frames), num_frames)
    return [raw_frames[i] for i in idx]


def midpoint_frames(frames: list) -> list:
    return [frames[len(frames) // 2]] * len(frames)


def shuffle_frames(frames: list, clip_id: str, backbone: str) -> list:
    """VM's C1 = shuffle pairs of consecutive frames as units (matches dfa_mass_delta_vm.py's
    preprocess_c1). TF's C = full-permutation shuffle of every frame (matches perturbation.py's
    apply_shuffle, used directly by dfa_mass_delta.py) — genuinely different conditions,
    not a stylistic difference, so no shared implementation here."""
    seed = _deterministic_seed(clip_id)
    if backbone == "vm":
        pairs = [(frames[i], frames[i + 1]) for i in range(0, len(frames), 2)]
        order = np.random.default_rng(seed).permutation(len(pairs)).tolist()
        return [f for i in order for f in pairs[i]]
    return apply_shuffle(frames, seed)


def extract_z(frames: list, model, processor, sae, dim_mean, model_flag: str, layer: int) -> torch.Tensor:
    model_cfg = MODEL_REGISTRY[model_flag]
    cls_offset = model_cfg["cls_offset"]
    pixel_values = processor(frames, return_tensors="pt")["pixel_values"].to(DEVICE)
    captured = {}

    def hook_fn(module, input, output):
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model_cfg["layer_getter"](model, layer).register_forward_hook(hook_fn)
    with torch.no_grad():
        model(pixel_values=pixel_values)
    handle.remove()

    patch_hidden = captured["hidden"][0, cls_offset:, :] - dim_mean
    _, z = sae.encode(patch_hidden.float())
    return z.detach().cpu()


def activation_map(z_grouped: torch.Tensor, feat: int, dec_sign: float) -> np.ndarray:
    """z_grouped: (num_positions, N_SPATIAL, dict_size), already in canonical
    position-major order via gather_by_position — NOT a flat (num_patch_tokens,
    dict_size) tensor. TF's native token order is patch-major/frame-minor, so a
    naive reshape here (as VM alone would tolerate) silently interleaves patches
    and frames — confirmed 06/08 from the checkerboard artifact it produced."""
    num_positions, n_spatial = z_grouped.shape[0], z_grouped.shape[1]
    spatial = int(n_spatial ** 0.5)
    signed = z_grouped[:, :, feat] * dec_sign
    return signed.numpy().reshape(num_positions, spatial, spatial)


def _overlay(frame: np.ndarray, patch: np.ndarray, norm, cmap, alpha: float = 0.45) -> np.ndarray:
    H, W = frame.shape[:2]
    ri = np.arange(H) * patch.shape[0] // H
    ci = np.arange(W) * patch.shape[1] // W
    heat = cmap(norm(patch[np.ix_(ri, ci)]))[:, :, :3]
    return (frame / 255.0 * (1 - alpha) + heat * alpha).clip(0, 1)


COND_LABELS = {"clip": "Clip", "R": "Real", "shuffle": "Shuffle", "A": "Still"}


def make_figure(maps: dict, frames_by_cond: dict, backbone: str, backbone_cfg: dict,
                 class_id: int, clip_id: str, feature_idx: int, bucket: str,
                 signed_R: float, signed_shuffle: float) -> plt.Figure:
    conditions = ["R", "shuffle", "A"]
    n_t = len(frames_by_cond["R"])
    fig, axes = plt.subplots(4, n_t, figsize=(n_t * 2, 9))
    fig.subplots_adjust(left=0.08, right=0.91, hspace=0.05, wspace=0.03, top=0.88)

    for col, frame in enumerate(frames_by_cond["R"]):
        axes[0, col].imshow(frame)
        axes[0, col].axis("off")
    mid_y = axes[0, 0].get_position().y0 + axes[0, 0].get_position().height / 2
    fig.text(0.01, mid_y, COND_LABELS["clip"], fontsize=11, fontweight="bold", va="center", ha="left")

    for row, cond in enumerate(conditions, start=1):
        act = maps[cond]
        display = frames_by_cond[cond]
        for col in range(n_t):
            axes[row, col].imshow(_overlay(display[col], act[col], maps["norm"], maps["cmap"]))
            axes[row, col].axis("off")
        mid_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.01, mid_y, COND_LABELS[cond], fontsize=11, fontweight="bold", va="center", ha="left")

    title = f"Feature {feature_idx} — {bucket} — Class {class_id} — Clip {clip_id} [{backbone.upper()}]"
    if INCLUDE_VALUES_IN_CAPTION:
        title += f"\nsigned_R={signed_R:+.4f}   signed_shuffle={signed_shuffle:+.4f}"
    fig.suptitle(title, fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=maps["cmap"], norm=maps["norm"])
    sm.set_array([])
    vmax = maps["norm"].vmax
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f"{-vmax:.1f}", "0", f"{vmax:.1f}"])
    cbar.set_label("signed activation", fontsize=9)
    return fig


def display_frames(sampled_frames: list, backbone: str) -> list:
    """8 display columns either way: VM's 16 sampled frames pair into 8 tubelets (show one
    frame per pair); TF's 8 sampled frames already are the 8 positions (show all of them)."""
    return sampled_frames[::2] if backbone == "vm" else sampled_frames


def load_clip_signed_vectors(backbone: str, clip_id: str) -> dict:
    """Per-condition DFA sign source — R/shuffle/A each get their own sign, not R's
    sign reused across all three (06/08: a fixed R-derived sign made Shuffle's color
    misleading whenever signed_shuffle itself had flipped)."""
    cfg = BACKBONE_CFG[backbone]
    shuffle_col = "signed_vec_C1" if backbone == "vm" else "signed_vec_C"
    df = pd.read_parquet(cfg["dfa_parquet"], columns=["clip_id", "signed_vec_R", shuffle_col, "signed_vec_A"])
    row = df[df.clip_id == clip_id].iloc[0]
    return {"R": np.asarray(row["signed_vec_R"]), "shuffle": np.asarray(row[shuffle_col]),
            "A": np.asarray(row["signed_vec_A"])}


def process_clip_group(backbone: str, class_id: int, clip_id: str, features: list,
                       model, processor, sae, dim_mean) -> None:
    cfg = BACKBONE_CFG[backbone]
    raw = load_raw_frames(clip_id, cfg["video_dir"])
    frames_R = sample_frames(raw, cfg["num_frames"])
    frames_shuffle = shuffle_frames(frames_R, clip_id, backbone)
    frames_A = midpoint_frames(frames_R)

    # gather_by_position converts each model's native flat token order into a shared
    # canonical (num_positions, N_SPATIAL, dict_size) layout — required for TF, harmless
    # reshape-only for VM (see activation_map's docstring for why this can't be skipped).
    z_R = gather_by_position(
        extract_z(frames_R, model, processor, sae, dim_mean, cfg["model_flag"], 7), cfg["model_flag"])
    z_shuffle = gather_by_position(
        extract_z(frames_shuffle, model, processor, sae, dim_mean, cfg["model_flag"], 7), cfg["model_flag"])
    z_A = gather_by_position(
        extract_z(frames_A, model, processor, sae, dim_mean, cfg["model_flag"], 7), cfg["model_flag"])

    frames_by_cond = {
        "R": display_frames(frames_R, backbone),
        "shuffle": display_frames(frames_shuffle, backbone),
        "A": display_frames(frames_A, backbone),
    }

    signed_vecs = load_clip_signed_vectors(backbone, clip_id)

    for feature_idx, bucket, signed_R, signed_shuffle in features:
        sign_R = float(np.sign(signed_vecs["R"][feature_idx])) or 1.0
        sign_shuffle = float(np.sign(signed_vecs["shuffle"][feature_idx])) or 1.0
        sign_A = float(np.sign(signed_vecs["A"][feature_idx])) or 1.0
        maps = {
            "R": activation_map(z_R, feature_idx, sign_R),
            "shuffle": activation_map(z_shuffle, feature_idx, sign_shuffle),
            "A": activation_map(z_A, feature_idx, sign_A),
        }
        vmax = float(np.percentile(np.abs(maps["R"]), 99)) or 1e-6  # guard degenerate all-zero R
        maps["norm"] = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        maps["cmap"] = cm.RdBu_r

        fig = make_figure(maps, frames_by_cond, backbone, cfg, class_id, clip_id,
                          feature_idx, bucket, signed_R, signed_shuffle)
        out_path = OUT_DIR / f"{backbone}_{class_id}_{clip_id}_{feature_idx}_{bucket}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path.name}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for backbone in ["vm", "tf"]:
        rows = [c for c in CANDIDATES if c[0] == backbone]
        if not rows:
            continue
        print(f"Loading {backbone} model + SAE...")
        model, processor, sae, dim_mean = load_model_and_sae(BACKBONE_CFG[backbone])

        groups: dict[tuple, list] = {}
        for _, class_id, clip_id, feature_idx, bucket, signed_R, signed_shuffle in rows:
            groups.setdefault((class_id, clip_id), []).append(
                (feature_idx, bucket, signed_R, signed_shuffle))

        for (class_id, clip_id), features in groups.items():
            print(f"[{backbone}] class {class_id}, clip {clip_id} ({len(features)} features)...")
            process_clip_group(backbone, class_id, clip_id, features, model, processor, sae, dim_mean)

    print("Done.")


if __name__ == "__main__":
    main()
