"""
Scaffold ablation extraction — logit-level, R + C1.

For each R-correct clip in the source parquet, runs 10 ablation targets under
R and C1 conditions using cached z from get_z_pixels. No backward pass.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_results_long.parquet    — one row per (clip, condition, target)
    ablation_targets.json            — TARGETS dict dump for manual inspection

Usage:
    uv run python src/stage3_analysis/run_ablation.py
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.dfa_engine import DFAEngine, _preprocess_clip
from stage3_analysis.ablation_targets import TARGETS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_flag":  "videomae",
    "device":      "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "mass_delta_dir": ROOT / "outputs/analysis/dfa_mass_delta_vm_c1",
    "video_dir":   os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "out_dir":     ROOT / "outputs/analysis/scaffold_ablation",
}


def _resolve_cfg(layer: int, job_label: str, sae_k: int) -> dict:
    """VM SAE checkpoint for the given layer/job_label — layer is a caller
    argument now (was hardcoded to 7), matching the extraction scripts'
    _resolve_cfg convention."""
    sae_path = ROOT / "outputs" / "sae" / f"sae_layer{layer}_job{job_label}.pt"
    dim_mean = ROOT / "outputs" / "sae" / f"layer{layer}_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean),
            "sae_k": sae_k, "layer": layer}


def _resolve_source_parquet(mass_delta_dir: Path, layer: int, job_label: str, sae_k: int) -> Path:
    """Per-clip signed-vector source — suffixed pattern preferred, falling back
    to the legacy unsuffixed file only for the original L7/job64/k64 baseline
    (matches scaffold_mass_pct.locate_source's precedent)."""
    suffixed = mass_delta_dir / f"dfa_mass_delta_vm_c1_l{layer}_job{job_label}_k{sae_k}.parquet"
    if suffixed.exists():
        return suffixed
    legacy = mass_delta_dir / "dfa_mass_delta_vm_c1.parquet"
    if (layer, job_label, sae_k) == (7, "64", 64) and legacy.exists():
        return legacy
    raise FileNotFoundError(f"No mass-delta source parquet found: {suffixed}")


def load_clips(cfg: dict) -> list[tuple[str, int, str, Path]]:
    df = pd.read_parquet(cfg["source_parquet"])
    video_dir = Path(cfg["video_dir"])
    result = []
    for _, row in df.iterrows():
        path = video_dir / f"{row['clip_id']}.webm"
        if path.exists():
            result.append((str(row["clip_id"]), int(row["class_id"]), row["sl_label"], path))
    log.info(f"  {len(result):,} clips found  ({len(df) - len(result)} missing on disk)")
    return result


def preprocess_c1(clip_path: Path, clip_id: str, num_frames: int, processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n       = len(frames)
    idx     = torch.linspace(0, n - 1, num_frames).long().tolist()
    sampled = [frames[i] for i in idx]
    pairs   = [(sampled[i], sampled[i + 1]) for i in range(0, num_frames, 2)]
    order   = np.random.default_rng(int(clip_id) % 2**32).permutation(len(pairs)).tolist()
    result  = [f for i in order for f in pairs[i]]
    return processor(result, return_tensors="pt")["pixel_values"].to(device)


def run_clip(
    engine: DFAEngine, clip_id: str, class_id: int,
    sl_label: str, clip_path: Path, device: str,
) -> list[dict]:
    pv_r  = _preprocess_clip(clip_path, engine._num_frames, engine._processor, device)
    pv_c1 = preprocess_c1(clip_path, clip_id, engine._num_frames, engine._processor, device)
    rows  = []
    for cond, pv in [("R", pv_r), ("C1", pv_c1)]:
        z_cache = engine.get_z_pixels(pv)
        base_logit, _, _, base_all_logits = engine.run_ablated(pv, class_id, [], z_cache)
        for target_name, indices in TARGETS.items():
            abl_logit, pred, correct, abl_all_logits = engine.run_ablated(pv, class_id, indices, z_cache)
            rows.append({
                "clip_id":                clip_id,
                "class_id":               class_id,
                "sl_label":               sl_label,
                "perturbation_condition": cond,
                "ablation_target":        target_name,
                "baseline_logit":         base_logit,
                "ablated_logit":          abl_logit,
                "delta":                  base_logit - abl_logit,
                "predicted_class_ablated": pred,
                "correct_ablated":        correct,
                "baseline_all_logits":    base_all_logits,
                "ablated_all_logits":     abl_all_logits,
            })
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--job-label", type=str, default="64")
    parser.add_argument("--sae-k", type=int, default=64)
    parser.add_argument("--run-tag", type=str, default=None,
                         help="defaults to l{layer}_job{job-label}_k{sae-k} if omitted")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dry_run = args.dry_run
    resolved = _resolve_cfg(args.layer, args.job_label, args.sae_k)
    source_parquet = _resolve_source_parquet(CFG["mass_delta_dir"], args.layer, args.job_label, args.sae_k)
    run_tag = args.run_tag or f"l{args.layer}_job{args.job_label}_k{args.sae_k}"
    cfg = {**CFG, **resolved, "source_parquet": source_parquet, "run_tag": run_tag}
    out_dir: Path = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "ablation_targets.json").write_text(json.dumps(TARGETS, indent=2))
    log.info(f"Targets ({len(TARGETS)}): {list(TARGETS.keys())}")

    clips = load_clips(cfg)
    if dry_run:
        clips = clips[:10]
        log.info("DRY RUN — 10 clips only")

    all_rows  = []
    clip_times = []
    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"], sae_k=cfg["sae_k"]) as engine:
        for i, (clip_id, class_id, sl_label, clip_path) in enumerate(clips):
            t0 = time.time()
            try:
                all_rows.extend(run_clip(engine, clip_id, class_id, sl_label, clip_path, cfg["device"]))
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}")
            elapsed = time.time() - t0
            clip_times.append(elapsed)
            log.info(f"[{i+1}/{len(clips)}] clip {clip_id}  {elapsed:.1f}s  rows: {len(all_rows):,}")

    df = pd.DataFrame(all_rows)
    tag      = "dry_run" if dry_run else cfg["run_tag"]
    out_path = out_dir / f"ablation_results_long_{tag}.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"  {len(df):,} rows → {out_path}")
    if dry_run and clip_times:
        mean_s = sum(clip_times) / len(clip_times)
        log.info(f"  Mean {mean_s:.1f}s/clip → full run estimate: {3558 * mean_s / 3600:.1f} hours")


if __name__ == "__main__":
    main()
