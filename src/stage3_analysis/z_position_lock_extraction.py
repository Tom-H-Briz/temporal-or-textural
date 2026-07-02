"""
Activation-based position lock extraction — z from get_z(), no backward pass.

For each clip, computes per-tubelet SAE activation (z, not DFA) for all features.
Accumulates per-clip share statistics per class, same corrected metric as the
fixed dfa_per_tubelet_mass.py (share computed per clip then averaged, not
aggregate-then-share).

R condition only — answers "where does this feature fire", not "where does it
causally contribute". No R-correct gate.

Token ordering: temporal-major — confirmed at feature_vis_group_vm.py:176.
tubelet_idx = token_idx // 196

Output: outputs/analysis/z_position_lock/z_position_lock_scores.csv

Usage:
    uv run python src/stage3_analysis/z_position_lock_extraction.py
"""

import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

NUM_TUBELETS = 8
DICT_SIZE    = 6144

CFG = {
    "model_flag":      "videomae",
    "layer":           7,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/z_position_lock"),
}


def _resolve_cfg() -> dict:
    sae_path = ROOT / "outputs" / "sae" / "sae_layer7_job64.pt"
    dim_mean = ROOT / "outputs" / "sae" / "layer7_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"VM SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": 64}


def load_clips(cfg: dict) -> list[tuple[str, int, Path]]:
    label_map, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    result = []
    for c in clips:
        cid = label_map.get(_strip_brackets(c["template"]))
        if cid not in DFA_CLASSES:
            continue
        path = video_dir / f"{c['id']}.webm"
        if path.exists():
            result.append((str(c["id"]), cid, path))
    log.info(f"  {len(result):,} clips across {len(DFA_CLASSES)} classes")
    return result


def main() -> None:
    resolved = _resolve_cfg()
    cfg      = CFG
    out_dir  = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = {int(r["class_id"]): r["category"]
              for _, r in pd.read_csv(cfg["sl_csv_path"]).iterrows()}
    clips  = load_clips(cfg)

    # Accumulators — same per-clip-first convention as fixed dfa_per_tubelet_mass.py
    running_share_sum  = defaultdict(lambda: torch.zeros(DICT_SIZE))
    tubelet_occurrence = defaultdict(lambda: torch.zeros(NUM_TUBELETS, DICT_SIZE))
    running_abs_sum    = defaultdict(lambda: torch.zeros(NUM_TUBELETS, DICT_SIZE))
    running_count      = defaultdict(int)

    log.info(f"Scanning {len(clips):,} clips (get_z, forward pass only)…")
    with DFAEngine(cfg["model_flag"], resolved["sae_path"], resolved["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=resolved["sae_k"]) as engine:

        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            try:
                z = engine.get_z(clip_path)   # (1568, dict_size)
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}"); continue

            # Per-tubelet z activation: (8, dict_size) — move to CPU for accumulation
            tubelet_z = z.reshape(NUM_TUBELETS, 196, DICT_SIZE).sum(dim=1).cpu()

            col_sum    = tubelet_z.sum(dim=0)                      # (dict_size,)
            active     = col_sum > 1e-8                            # (dict_size,) bool — feature fired this clip
            col_sum_c  = col_sum.clamp(min=1e-10)
            col_max, col_argmax = tubelet_z.max(dim=0)            # (dict_size,)

            running_share_sum[class_id]  += col_max / col_sum_c
            running_abs_sum[class_id]    += tubelet_z
            # Only vote for a tubelet position when the feature actually fired
            active_argmax = col_argmax * active.long()            # zero out inactive — harmless since scatter is masked
            tubelet_occurrence[class_id].scatter_add_(
                0, active_argmax.unsqueeze(0), active.float().unsqueeze(0)
            )
            running_count[class_id] += 1

            if (i + 1) % 500 == 0:
                log.info(f"  [{i+1}/{len(clips)}] scanned")

    log.info("Building output CSV…")
    rows = []
    for class_id in sorted(running_count):
        n     = running_count[class_id]
        label = sl_map.get(class_id, "unlabelled")

        mean_share = (running_share_sum[class_id] / n).numpy()    # (dict_size,)
        mean_abs   = (running_abs_sum[class_id]   / n).numpy()    # (8, dict_size)
        occ        = tubelet_occurrence[class_id].numpy()          # (8, dict_size)
        mode_t     = occ.argmax(axis=0)                           # (dict_size,)
        frac_mode  = occ.max(axis=0) / n                          # (dict_size,)
        total_abs  = mean_abs.sum(axis=0)                         # (dict_size,)
        top_abs    = mean_abs.max(axis=0)                         # (dict_size,)

        for feat in range(DICT_SIZE):
            rows.append({
                "class_id":                class_id,
                "sl_label":               label,
                "feature_idx":            feat,
                "n_clips":                n,
                "mean_per_clip_share_R":  float(mean_share[feat]),
                "mode_tubelet_R":         int(mode_t[feat]),
                "frac_clips_matching_mode_R": float(frac_mode[feat]),
                "total_abs_R":            float(total_abs[feat]),
                "top_abs_R":              float(top_abs[feat]),
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / "z_position_lock_scores.csv"
    df.to_csv(out_path, index=False)
    log.info(f"  → {out_path}  ({len(df):,} rows)")
    log.info("Done.")


if __name__ == "__main__":
    main()
