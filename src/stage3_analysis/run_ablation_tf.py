"""
TF sign-flip feature ablation — extraction, logit-level, R + C.

Top-5/Top-12 TF-SSv2 layer-7 sign-flip-concentrated features (CC brief
06/08/26): ablates them singly and jointly across the full 35-class SL
population under R and C (full-permutation shuffle) to test whether they
behave as a coordinated flip mechanism.

Outputs (outputs/analysis/scaffold_ablation/):
    tf_signflip_ablation_results_long.parquet — one row per (clip, condition, target)

Usage:
    uv run python src/stage3_analysis/run_ablation_tf.py
    uv run python src/stage3_analysis/run_ablation_tf.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import av
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))

from stage3_analysis.dfa_engine import DFAEngine, _preprocess_clip
from perturbation import apply_shuffle
from ToT_utils import _strip_brackets, load_metadata, resolve_sae_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_flag": "timesformer",
    "layer": 7,
    "sae_k": 64,
    "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "manifest_path": str(ROOT / "outputs/Laura_SL/manifest_SL_subset.json"),
    "sl_csv_path": str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "out_dir": ROOT / "outputs/analysis/scaffold_ablation",
}

# Top-5/Top-12 TF-SSv2 L7 sign-flip-concentrated features (CC brief 06/08/26,
# k64/x8 SAE — indices are dictionary-specific, not comparable to k128/x8).
TARGETS = {
    "single_3029": [3029], "single_1517": [1517], "single_2090": [2090],
    "single_2057": [2057], "single_2156": [2156],
    "TOP5":  [3029, 1517, 2090, 2057, 2156],
    "TOP12": [3029, 1517, 2090, 2057, 2156, 1588, 4590, 3813, 6029, 622, 1371, 4134],
}


def build_sl_label_map(cfg: dict) -> dict[int, str]:
    df = pd.read_csv(cfg["sl_csv_path"])
    return {int(row["class_id"]): row["category"] for _, row in df.iterrows()}


def load_clips(cfg: dict) -> list[tuple[str, int, str, Path]]:
    label_map, _, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    sl_map = build_sl_label_map(cfg)
    video_dir = Path(cfg["video_dir"])
    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)
    result = []
    for sl_label, entries in manifest.items():
        for entry in entries:
            cid = label_map.get(_strip_brackets(entry["template"]))
            path = video_dir / f"{entry['id']}.webm"
            if cid is not None and path.exists():
                result.append((str(entry["id"]), cid, sl_map.get(cid, sl_label), path))
    log.info(f"  {len(result):,} clips from SL manifest")
    return result


def preprocess_c(clip_path: Path, clip_id: str, num_frames: int, processor, device: str) -> torch.Tensor:
    """TF's real shuffle condition (full-permutation, matches dfa_mass_delta.py's
    preprocess_c) — distinct from VM's paired-frame C1, so labeled "C" not "C1"."""
    container = av.open(str(clip_path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, int(clip_id) % 2**32)
    n = len(frames)
    idx = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def run_clip(engine: DFAEngine, clip_id: str, class_id: int, sl_label: str,
            clip_path: Path, device: str) -> list[dict]:
    pv_r = _preprocess_clip(clip_path, engine._num_frames, engine._processor, device)
    z_r = engine.get_z_pixels(pv_r)
    base_logit_r, _, correct_r, _ = engine.run_ablated(pv_r, class_id, [], z_r)
    if not correct_r:
        return []

    pv_c = preprocess_c(clip_path, clip_id, engine._num_frames, engine._processor, device)
    rows = []
    for cond, pv, z_cache, base_logit in [("R", pv_r, z_r, base_logit_r),
                                           ("C", pv_c, engine.get_z_pixels(pv_c), None)]:
        if base_logit is None:
            base_logit, _, _, _ = engine.run_ablated(pv, class_id, [], z_cache)
        for target_name, indices in TARGETS.items():
            abl_logit, pred, correct, _ = engine.run_ablated(pv, class_id, indices, z_cache)
            rows.append({
                "clip_id": clip_id, "class_id": class_id, "sl_label": sl_label,
                "perturbation_condition": cond, "ablation_target": target_name,
                "baseline_logit": base_logit, "ablated_logit": abl_logit,
                "delta": base_logit - abl_logit,
                "predicted_class_ablated": pred, "correct_ablated": correct,
            })
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    resolved = resolve_sae_checkpoint(CFG["model_flag"], CFG["layer"],
                                      dataset_name="ssv2", sae_k=CFG["sae_k"])
    cfg = {**CFG, **resolved}
    out_dir: Path = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tf_signflip_ablation_targets.json").write_text(json.dumps(TARGETS, indent=2))
    log.info(f"Targets ({len(TARGETS)}): {list(TARGETS.keys())}")

    clips = load_clips(cfg)
    full_clip_count = len(clips)
    if args.dry_run:
        clips = clips[:10]
        log.info("DRY RUN — 10 clips only")

    all_rows, clip_times = [], []
    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"], sae_k=cfg["sae_k"],
                   dataset_name="ssv2") as engine:
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
    tag = "dry_run" if args.dry_run else "tf_signflip"
    out_path = out_dir / f"{tag}_ablation_results_long.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"  {len(df):,} rows → {out_path}")
    if args.dry_run and clip_times:
        mean_s = sum(clip_times) / len(clip_times)
        log.info(f"  Mean {mean_s:.1f}s/clip → full run ({full_clip_count} clips) "
                 f"estimate: {full_clip_count * mean_s / 3600:.1f} hours")


if __name__ == "__main__":
    main()
