"""
Median re-aggregation of ablation margin data — no new ablation run.

Reads existing per-clip margin CSVs and re-aggregates using median for
pct_change columns (mean is distorted by near-zero-margin clips).

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_median_sl_{tag}.csv          — sl_label pooled, median metrics
    ablation_median_class_{tag}.csv       — per class, median metrics
    ablation_median_class_comparison.csv  — clean7 vs all4 side by side, per class

Usage:
    uv run python src/stage3_analysis/ablation_median_summary.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT  = ROOT / "outputs/analysis/scaffold_ablation"

RUNS = [
    {"tag": "clean7_020726",        "target": "clean7"},
    {"tag": "cover_uncover_060726", "target": "all4"},
]

ACCURACY_PATH    = ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"
ACCURACY_THRESHOLD = 0.40

AGG = {
    "n_clips":                       ("clip_id",                     "count"),
    "median_margin_r1r2_unabl":      ("margin_r1r2_unabl",           "median"),
    "median_margin_r1r2_pct":        ("margin_r1r2_pct_change",      "median"),
    "median_margin_r1_mean25_pct":   ("margin_r1_mean25_pct_change", "median"),
    "mean_rank1_delta":              ("rank1_delta",                 "mean"),
    "flip_rate":                     ("flip",                        "mean"),
}


def summarise(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, as_index=False).agg(**AGG)


def main() -> None:
    acc = pd.read_csv(ACCURACY_PATH)
    valid_classes = set(acc.loc[acc["accuracy"] >= ACCURACY_THRESHOLD, "class_id"])
    dropped = set(acc["class_id"]) - valid_classes
    print(f"Accuracy filter (<{ACCURACY_THRESHOLD:.0%}): dropping {sorted(dropped)}")

    class_frames = {}

    for run in RUNS:
        tag = run["tag"]
        src = OUT / f"ablation_margin_sl_all_{tag}.csv"
        df  = pd.read_csv(src)
        df  = df[df["true_class"].isin(valid_classes)].copy()
        print(f"  {tag}: {len(df):,} clips after filter")

        sl  = summarise(df, ["sl_label"])
        cls = summarise(df, ["true_class", "sl_label"])

        sl.to_csv(OUT / f"ablation_median_sl_{tag}.csv", index=False)
        cls.to_csv(OUT / f"ablation_median_class_{tag}.csv", index=False)
        print(f"    → ablation_median_sl_{tag}.csv")
        print(f"    → ablation_median_class_{tag}.csv")
        print(sl.round(3).to_string(index=False))
        print()

        class_frames[run["target"]] = cls

    # Side-by-side class comparison
    c7  = class_frames["clean7"].rename(columns={
        "median_margin_r1r2_pct":      "c7_median_r1r2_pct",
        "median_margin_r1_mean25_pct": "c7_median_r1m25_pct",
        "flip_rate":                   "c7_flip_rate",
        "mean_rank1_delta":            "c7_mean_rank1_delta",
    })
    a4  = class_frames["all4"].rename(columns={
        "median_margin_r1r2_pct":      "a4_median_r1r2_pct",
        "median_margin_r1_mean25_pct": "a4_median_r1m25_pct",
        "flip_rate":                   "a4_flip_rate",
        "mean_rank1_delta":            "a4_mean_rank1_delta",
    })
    comp = c7.merge(a4[["true_class", "a4_median_r1r2_pct", "a4_median_r1m25_pct",
                         "a4_flip_rate", "a4_mean_rank1_delta"]], on="true_class")
    comp["delta_median_r1r2_pct"] = comp["a4_median_r1r2_pct"] - comp["c7_median_r1r2_pct"]
    comp = comp.sort_values("delta_median_r1r2_pct").reset_index(drop=True)

    comp.to_csv(OUT / "ablation_median_class_comparison.csv", index=False)
    print("=== Per-class comparison (sorted by delta, most-hit first) ===")
    print(comp[["true_class", "sl_label", "n_clips",
                "median_margin_r1r2_unabl",
                "c7_median_r1r2_pct", "a4_median_r1r2_pct",
                "delta_median_r1r2_pct", "c7_flip_rate", "a4_flip_rate"]
              ].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
