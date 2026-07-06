"""
SL-wide ablation margin analysis — clean7, R condition, full SL subset.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_margin_sl_all.csv   — one row per clip
    ablation_sl_summary.csv      — aggregated per sl_label

Usage:
    uv run python src/stage3_analysis/ablation_margin_sl.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_targets import TARGETS

CFG = {
    "source":    ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long.parquet",
    "out_dir":   ROOT / "outputs/analysis/scaffold_ablation",
    "target":    "all4",
    "condition": "R",
    "run_tag":   "cover_uncover_060726",   # change for each new run — stamps all output filenames
    "confound_threshold": 0.5,
}


def build_per_clip(df: pd.DataFrame) -> pd.DataFrame:
    base_mat = np.stack(df["baseline_all_logits"].to_numpy()).astype(np.float32)
    abl_mat  = np.stack(df["ablated_all_logits"].to_numpy()).astype(np.float32)
    N        = len(df)
    idx      = np.arange(N)

    ranked   = np.argsort(base_mat, axis=1)[:, ::-1]   # (N, 174)
    r1_cls   = ranked[:, 0];  r2_cls = ranked[:, 1]
    r2_5_idx = ranked[:, 1:5]                           # ranks 2–5

    r1_u = base_mat[idx, r1_cls];  r1_a = abl_mat[idx, r1_cls]
    r2_u = base_mat[idx, r2_cls];  r2_a = abl_mat[idx, r2_cls]
    m25_u = base_mat[idx[:, None], r2_5_idx].mean(axis=1)
    m25_a = abl_mat[idx[:, None],  r2_5_idx].mean(axis=1)

    mar12_u = r1_u - r2_u;  mar12_a = r1_a - r2_a
    mar25_u = r1_u - m25_u; mar25_a = r1_a - m25_a

    return pd.DataFrame({
        "clip_id":                       df["clip_id"].values,
        "true_class":                    df["class_id"].values.astype(int),
        "sl_label":                      df["sl_label"].values,
        "rank1_logit_unabl":             r1_u,
        "rank1_logit_abl":               r1_a,
        "rank1_delta":                   r1_a - r1_u,
        "rank2_class":                   r2_cls.astype(int),
        "rank2_logit_unabl":             r2_u,
        "rank2_logit_abl":               r2_a,
        "margin_r1r2_unabl":             mar12_u,
        "margin_r1r2_abl":               mar12_a,
        "margin_r1r2_pct_change":        (mar12_a - mar12_u) / np.abs(mar12_u) * 100,
        "mean25_logit_unabl":            m25_u,
        "mean25_logit_abl":              m25_a,
        "margin_r1_mean25_unabl":        mar25_u,
        "margin_r1_mean25_abl":          mar25_a,
        "margin_r1_mean25_pct_change":   (mar25_a - mar25_u) / np.abs(mar25_u) * 100,
        "flip":                          (mar12_a < 0).astype(int),
    })


def build_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    grp = df.groupby("sl_label")
    summary = grp.agg(
        n_clips                          =("clip_id",                     "count"),
        mean_margin_r1r2_unabl           =("margin_r1r2_unabl",           "mean"),
        sd_margin_r1r2_unabl             =("margin_r1r2_unabl",           "std"),
        mean_margin_r1r2_pct_change      =("margin_r1r2_pct_change",      "mean"),
        sd_margin_r1r2_pct_change        =("margin_r1r2_pct_change",      "std"),
        mean_margin_r1_mean25_pct_change =("margin_r1_mean25_pct_change", "mean"),
        sd_margin_r1_mean25_pct_change   =("margin_r1_mean25_pct_change", "std"),
        n_flips                          =("flip",                        "sum"),
        flip_rate                        =("flip",                        "mean"),
    ).reset_index()

    vals = summary.set_index("sl_label")["mean_margin_r1r2_unabl"]
    if {"temporal", "static"}.issubset(vals.index):
        gap = abs(vals["temporal"] - vals["static"])
        summary["baseline_margin_gap"] = gap
        if gap > threshold:
            print(f"CONFOUND WARNING: baseline margins differ by {gap:.3f} logit units (threshold {threshold})")
        else:
            print(f"Confound check OK: baseline margin gap = {gap:.3f}")

    return summary


def main() -> None:
    out_dir: Path = CFG["out_dir"]
    df = pd.read_parquet(CFG["source"])
    df = df[(df["perturbation_condition"] == CFG["condition"]) &
            (df["ablation_target"]        == CFG["target"])].copy()
    print(f"  {len(df):,} clips  |  sl_labels: {df['sl_label'].value_counts().to_dict()}")

    tag = CFG["run_tag"]
    per_clip = build_per_clip(df)
    per_clip.to_csv(out_dir / f"ablation_margin_sl_all_{tag}.csv", index=False)
    print(f"  {len(per_clip):,} rows → ablation_margin_sl_all_{tag}.csv")

    summary = build_summary(per_clip, CFG["confound_threshold"])
    summary.to_csv(out_dir / f"ablation_sl_summary_{tag}.csv", index=False)

    print()
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
