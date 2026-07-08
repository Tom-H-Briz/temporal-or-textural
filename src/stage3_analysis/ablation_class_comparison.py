"""
Per-class ablation margin comparison — clean7 vs all4, R condition.

Reads both run parquets and produces a wide-format CSV comparing how much
each class's rank-1 margin contracted under each set of ablated features.

Output: outputs/analysis/scaffold_ablation/ablation_class_comparison.csv

Usage:
    uv run python src/stage3_analysis/ablation_class_comparison.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs/analysis/scaffold_ablation"

RUNS = [
    {"label": "clean7", "parquet": OUT / "ablation_results_long_clean7_020726.parquet",     "target": "clean7"},
    {"label": "all4",   "parquet": OUT / "ablation_results_long_cover_uncover_060726.parquet", "target": "all4"},
]

BAD_CLASSES = {38, 97, 160}


def build_per_clip(df: pd.DataFrame) -> pd.DataFrame:
    base_mat = np.stack(df["baseline_all_logits"].to_numpy()).astype(np.float32)
    abl_mat  = np.stack(df["ablated_all_logits"].to_numpy()).astype(np.float32)
    N, idx   = len(df), np.arange(len(df))
    ranked   = np.argsort(base_mat, axis=1)[:, ::-1]
    r1_cls, r2_cls = ranked[:, 0], ranked[:, 1]
    r2_5_idx = ranked[:, 1:5]
    r1_u = base_mat[idx, r1_cls]; r1_a = abl_mat[idx, r1_cls]
    r2_u = base_mat[idx, r2_cls]; r2_a = abl_mat[idx, r2_cls]
    m25_u = base_mat[idx[:, None], r2_5_idx].mean(axis=1)
    m25_a = abl_mat[idx[:, None],  r2_5_idx].mean(axis=1)
    mar12_u = r1_u - r2_u; mar12_a = r1_a - r2_a
    mar25_u = r1_u - m25_u; mar25_a = r1_a - m25_a
    return pd.DataFrame({
        "clip_id":                     df["clip_id"].values,
        "class_id":                    df["class_id"].values.astype(int),
        "sl_label":                    df["sl_label"].values,
        "rank1_delta":                 r1_a - r1_u,
        "margin_r1r2_pct_change":      (mar12_a - mar12_u) / np.abs(mar12_u) * 100,
        "margin_r1_mean25_pct_change": (mar25_a - mar25_u) / np.abs(mar25_u) * 100,
        "flip":                        (mar12_a < 0).astype(int),
    })


def class_summary(per_clip: pd.DataFrame) -> pd.DataFrame:
    return per_clip.groupby(["class_id", "sl_label"]).agg(
        n_clips                      =("clip_id",                     "count"),
        mean_rank1_delta             =("rank1_delta",                 "mean"),
        mean_margin_r1r2_pct         =("margin_r1r2_pct_change",      "mean"),
        mean_margin_r1_mean25_pct    =("margin_r1_mean25_pct_change", "mean"),
        flip_rate                    =("flip",                        "mean"),
    ).reset_index()


def main() -> None:
    summaries = {}
    for run in RUNS:
        df = pd.read_parquet(run["parquet"])
        df = df[(df["perturbation_condition"] == "R") &
                (df["ablation_target"] == run["target"]) &
                (~df["class_id"].isin(BAD_CLASSES))].copy()
        print(f"  {run['label']}: {len(df):,} clips")
        per_clip = build_per_clip(df)
        summaries[run["label"]] = class_summary(per_clip)

    base = summaries["clean7"]
    comp = summaries["all4"][["class_id", "mean_rank1_delta", "mean_margin_r1r2_pct",
                               "mean_margin_r1_mean25_pct", "flip_rate"]]
    merged = base.merge(comp, on="class_id", suffixes=("_clean7", "_all4"))
    merged["delta_margin_r1r2_pct"]   = merged["mean_margin_r1r2_pct_all4"]   - merged["mean_margin_r1r2_pct_clean7"]
    merged["delta_margin_r1m25_pct"]  = merged["mean_margin_r1_mean25_pct_all4"] - merged["mean_margin_r1_mean25_pct_clean7"]
    merged["delta_flip_rate"]         = merged["flip_rate_all4"] - merged["flip_rate_clean7"]
    merged = merged.sort_values("delta_margin_r1r2_pct").reset_index(drop=True)

    out = OUT / "ablation_class_comparison.csv"
    merged.to_csv(out, index=False)
    print(f"\n→ {out}")
    print(merged[["class_id", "sl_label", "n_clips",
                  "mean_margin_r1r2_pct_clean7", "mean_margin_r1r2_pct_all4",
                  "delta_margin_r1r2_pct", "flip_rate_clean7", "flip_rate_all4"]
                ].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
