"""
DFA ablation validation — TF, 32 SL classes, layers [3,5,7,9].

For each sampled clip: reads stored top-K DFA scores from parquet,
runs K ablation forward passes (no backward), computes Pearson r between
DFA score and actual logit drop. Validates that DFA attribution predicts
causal importance.

Usage:
    SAE_LAYER=7 uv run python notebooks/dfa_ablation_tf.py
    SAE_LAYER=7 N_CLIPS=50 TOP_K=50 uv run python notebooks/dfa_ablation_tf.py
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.dfa_engine import DFAEngine, _preprocess_clip

import pandas as pd

_LAYER = int(os.environ.get("SAE_LAYER", 7))

CFG = {
    "model_flag":      "timesformer",
    "layer":           _LAYER,
    "n_clips":         int(os.environ.get("N_CLIPS", 100)),
    "top_k":           int(os.environ.get("TOP_K",   50)),
    "seed":            42,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "dfa_dir":         str(ROOT / "outputs/dfa"),
    "output_dir":      str(ROOT / "outputs/dfa"),
}


def _resolve(cfg: dict) -> dict:
    sae_dir = ROOT / "outputs" / "sae"
    layer   = cfg["layer"]
    matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 checkpoint for layer {layer}: {matches}")
    dim_mean = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    parquets = sorted(Path(cfg["dfa_dir"]).glob(f"dfa_tf_k*_l{layer}_sl32_*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No DFA parquet found for layer {layer} in {cfg['dfa_dir']}")
    ckpt  = torch.load(matches[0], map_location="cpu", weights_only=True)
    sae_k = ckpt.get("sae_k")
    if sae_k is None:
        raise ValueError(f"sae_k not in checkpoint — run patch_checkpoints_add_sae_k.py first")
    return {"sae_path": str(matches[0]), "dim_mean_path": str(dim_mean),
            "parquet_path": str(parquets[-1]), "sae_k": sae_k}


def ablate_clip(engine: DFAEngine, row: pd.Series, cfg: dict) -> dict | None:
    clip_path = Path(cfg["video_dir"]) / f"{row['clip_id']}.webm"
    if not clip_path.exists():
        return None

    z = engine.get_z(clip_path)  # (num_patch_tokens, dict_size) — no backward
    pixel_values = _preprocess_clip(
        clip_path, engine._num_frames, engine._processor, cfg["device"]
    )

    engine._z_override = z
    with torch.no_grad():
        baseline_logit = engine._model(pixel_values=pixel_values).logits[0, int(row["class_id"])].item()
    engine._z_override = None

    top_k   = cfg["top_k"]
    indices = np.array(row["top200_indices"])[:top_k]
    dfa_abs = np.array(row["top200_dfa_abs"])[:top_k]
    valid   = indices >= 0
    indices, dfa_abs = indices[valid], dfa_abs[valid]
    if len(indices) < 5:
        return None

    drops = []
    for feat_idx in indices:
        z_abl = z.clone()
        z_abl[:, int(feat_idx)] = 0.0
        engine._z_override = z_abl
        with torch.no_grad():
            abl_logit = engine._model(pixel_values=pixel_values).logits[0, int(row["class_id"])].item()
        engine._z_override = None
        drops.append(baseline_logit - abl_logit)

    r, _ = pearsonr(dfa_abs, drops)
    return {"clip_id": row["clip_id"], "class_id": int(row["class_id"]),
            "correct": bool(row["correct"]), "pearson_r": float(r),
            "mean_logit_drop": float(np.mean(drops)), "n_ablated": len(indices)}


def main() -> None:
    resolved = _resolve(CFG)
    cfg      = {**CFG, **resolved}
    layer    = cfg["layer"]
    print(f"Layer {layer}  device={cfg['device']}  n_clips={cfg['n_clips']}  top_k={cfg['top_k']}")

    df   = pd.read_parquet(cfg["parquet_path"])
    correct = df[df["correct"]].reset_index(drop=True)
    sample  = correct.sample(n=min(cfg["n_clips"], len(correct)), random_state=cfg["seed"])
    print(f"  {len(correct):,} correct clips available → sampling {len(sample)}")

    out_csv = Path(cfg["output_dir"]) / f"dfa_ablation_tf_l{layer}.csv"
    results = []

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=layer, device=cfg["device"]) as engine:
        for i, (_, row) in enumerate(sample.iterrows()):
            rec = ablate_clip(engine, row, cfg)
            if rec is None:
                continue
            results.append(rec)
            if (i + 1) % 10 == 0:
                rs = [r["pearson_r"] for r in results]
                print(f"  [{i+1}/{len(sample)}]  median r={np.median(rs):.3f}  mean r={np.mean(rs):.3f}")

    rs = [r["pearson_r"] for r in results]
    print(f"\nLayer {layer} summary ({len(results)} clips):")
    print(f"  mean r={np.mean(rs):.4f}  median r={np.median(rs):.4f}  "
          f"frac r>0.3: {np.mean(np.array(rs)>0.3):.3f}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"  Saved → {out_csv}")


if __name__ == "__main__":
    main()
