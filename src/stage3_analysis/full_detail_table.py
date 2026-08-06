"""
Full-population per-instance detail cache — CC brief 06/08 (increase/decrease
bucket characterisation).

Materialises ssv2_{vm,tf}_full_detail.parquet: one row per (clip, top-10
feature) across ALL SL-eligible classes (33 VM / 31 TF), not just one class
at a time — class_feature_breakdown.py computes the same rows but only ever
for a single --class-id, never cached across the population. Reuses
top10_detail() unchanged as the single source of truth for bucket math.

Usage:
    uv run python src/stage3_analysis/full_detail_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.clip_shuffle_disruption import CONFIGS, eligible_classes, top10_detail

OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"


def build_full_detail(backbone: str) -> pd.DataFrame:
    cfg = CONFIGS[f"ssv2_{backbone}"]
    dfa_df = pd.read_parquet(cfg["dfa_parquet"])
    classes = eligible_classes(dfa_df, cfg["r_acc_csv"])
    clips = dfa_df[dfa_df["class_id"].isin(classes)]

    rows = []
    for _, clip in clips.iterrows():
        r = np.asarray(clip["signed_vec_R"], dtype=np.float32)
        c = np.asarray(clip[cfg["shuffle_col"]], dtype=np.float32)
        detail = top10_detail(r, c)
        detail["clip_id"] = clip["clip_id"]
        detail["class_id"] = clip["class_id"]
        detail["correct_under_shuffle"] = bool(clip[cfg["correct_col"]])
        rows.append(detail)
    out = pd.concat(rows, ignore_index=True).rename(columns={"feature_idx": "feature_id"})
    print(f"  [{backbone}] {len(classes)} classes, {len(clips)} clips, {len(out)} instance rows")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for backbone in ("vm", "tf"):
        print(f"Processing {backbone}...")
        df = build_full_detail(backbone)
        path = OUT_DIR / f"ssv2_{backbone}_full_detail.parquet"
        df.to_parquet(path, index=False)
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
