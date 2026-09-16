"""
Synthetic-input probe: do position-locked features fire on content-free video?

Hypothesis (design-chat ledger): position-locked features feed off the temporal
positional encoding. Test: feed VideoMAE (ssv2 + kinetics400) synthetic videos
whose 16 frames are ALL the same content — white (255), black (0), or iid
gaussian noise — so every tubelet position carries identical content and the
positional encoding is the only thing that differs across the 8 positions.
Then measure, per scaffold member feature (the strict >=0.90-share,
every-clip-matching position-locked set), whether its per-position z still
peaks at its locked tubelet.

Interpretation:
  fires at locked position on constant content  -> lock is position-encoding
                                                   driven (hypothesis supported)
  uniform across positions / inactive           -> lock is a content-position
                                                   correlation on real data

Metric: engine.get_z_pixels (label-free) -> gather_by_position(z).sum(patches)
— the identical raw-activation metric and code path the locks were computed
with. DFAEngine is used unmodified, including its threshold-init behaviour, so
numbers are apples-to-apples with the existing position-lock parquets.

Outputs (outputs/analysis/pos_lock_synthetic/):
  pos_lock_synthetic_probe.csv   long: dataset, layer, feature_idx,
                                 locked_position, content, seed, active,
                                 z_at_locked, share_at_locked, argmax_position,
                                 fired_at_locked
  stdout summary                 fire-rate + mean share per (dataset, layer, content)

Usage: uv run python notebooks/pos_lock_synthetic_probe.py   (CPU, a few minutes)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "stage3_analysis"))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import (
    CHECKPOINT_REGISTRY, MODEL_REGISTRY, gather_by_position, get_processor,
    resolve_sae_checkpoint,
)
from dfa_engine import DFAEngine

CFG = {
    "model_name":   "videomae",
    "sae_k":        64,
    "job_label":    "7ep",
    # every k64 config with scaffold members (L9 has none on either dataset)
    "configs": [("ssv2", 3), ("ssv2", 5), ("ssv2", 7),
                ("kinetics400", 3), ("kinetics400", 5), ("kinetics400", 7)],
    "n_frames":      16,
    "frame_size":    224,
    "noise_mean":    128,
    "noise_std":     64,
    "n_noise_seeds": 16,  # white/black are deterministic; noise gets a rate, not a coin-flip
    "scaffold_dir":  ROOT / "outputs" / "analysis" / "scaffold_selection",
    "output_dir":    ROOT / "outputs" / "analysis" / "pos_lock_synthetic",
}


def make_frames(content: str, seed: int) -> list[np.ndarray]:
    """16 uint8 (H, W, 3) frames of one content type. Noise is iid per pixel per
    frame (temporally varying), clipped to the valid range, deterministically seeded."""
    h = w = CFG["frame_size"]
    if content == "white":
        return [np.full((h, w, 3), 255, dtype=np.uint8) for _ in range(CFG["n_frames"])]
    if content == "black":
        return [np.full((h, w, 3), 0, dtype=np.uint8) for _ in range(CFG["n_frames"])]
    rng = np.random.default_rng(seed)
    return [
        np.clip(rng.normal(CFG["noise_mean"], CFG["noise_std"], (h, w, 3)), 0, 255).astype(np.uint8)
        for _ in range(CFG["n_frames"])
    ]


def scaffold_members(dataset_name: str, layer: int) -> list[tuple[int, int]]:
    """(feature_idx, locked_position) for the strict members of one config.
    z_position is the lock under the z metric this probe measures (equal to
    dfa_position for members, by the selection's own agreement assert)."""
    suffix = "VM_K400" if dataset_name == "kinetics400" else "VM"
    path = CFG["scaffold_dir"] / f"L{layer}_x8k64_{suffix}.csv"
    df = pd.read_csv(path)
    members = df[df["status"] == "member"]
    return list(zip(members["feature_idx"].astype(int), members["z_position"].astype(int)))


def synthetic_videos() -> list[tuple[str, int]]:
    """(content, seed) per video — white/black once each (deterministic), noise x N seeds."""
    return [("white", 0), ("black", 0)] + [("noise", s) for s in range(CFG["n_noise_seeds"])]


def position_z(engine: DFAEngine, pixel_values: torch.Tensor) -> torch.Tensor:
    """(8, dict_size) per-tubelet z summed over the 196 patches — the pipeline's
    own raw-activation aggregation (identical to position_lock_extraction's path)."""
    z = engine.get_z_pixels(pixel_values)  # (1568, dict_size), label-free
    return gather_by_position(z, CFG["model_name"]).sum(dim=1)


def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = MODEL_REGISTRY[CFG["model_name"]]
    rows = []

    for dataset_name, layer in CFG["configs"]:
        members = scaffold_members(dataset_name, layer)
        if not members:
            print(f"{dataset_name} L{layer}: no scaffold members, skipping")
            continue
        resolved = resolve_sae_checkpoint(CFG["model_name"], layer, dataset_name,
                                          CFG["sae_k"], CFG["job_label"])
        processor = get_processor(model_cfg, CHECKPOINT_REGISTRY[(CFG["model_name"], dataset_name)])
        with DFAEngine(CFG["model_name"], resolved["sae_path"], resolved["dim_mean_path"],
                       layer=layer, device=device, sae_k=resolved["sae_k"],
                       dataset_name=dataset_name) as engine:
            for content, seed in synthetic_videos():
                pv = processor(make_frames(content, seed),
                               return_tensors="pt")["pixel_values"].to(device)
                per_pos = position_z(engine, pv)  # (8, dict_size)
                for feat, locked in members:
                    col = per_pos[:, feat]
                    total = float(col.sum())
                    active = total > 1e-8  # the pipeline's own activity gate
                    rows.append({
                        "dataset": dataset_name, "layer": layer,
                        "feature_idx": feat, "locked_position": locked,
                        "content": content, "seed": seed, "active": active,
                        "z_at_locked": round(float(col[locked]), 4),
                        "share_at_locked": round(float(col[locked] / total), 4) if active else 0.0,
                        "argmax_position": int(col.argmax()) if active else -1,
                        "fired_at_locked": bool(active and int(col.argmax()) == locked),
                    })
        print(f"{dataset_name} L{layer}: {len(members)} members x {len(synthetic_videos())} videos")


    df = pd.DataFrame(rows)
    out_path = out_dir / "pos_lock_synthetic_probe.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df)} rows)")

    summary = (df.groupby(["dataset", "layer", "content"])
                 .agg(fire_rate=("fired_at_locked", "mean"),
                      active_rate=("active", "mean"),
                      mean_share=("share_at_locked", "mean"))
                 .round(4).reset_index())
    print("\nPosition-locked features on synthetic constant content:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
