"""
TF sign-flip ablation summary — aggregates run_ablation_tf.py's output.

Outputs (outputs/analysis/scaffold_ablation/):
    tf_signflip_ablation_summary.csv    — static/temporal delta + ratio, K400-comparable shape
    tf_signflip_ablation_additivity.csv — sum(5 singletons) - TOP5, per condition
    tf_signflip_ablation_class_detail.csv — per-class detail, TOP5 + TOP12, R condition

Usage:
    uv run python src/stage3_analysis/ablation_summary_tf_signflip.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

CFG = {
    "results_path": ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_results_long.parquet",
    "out_dir": ROOT / "outputs/analysis/scaffold_ablation",
    "layer": 7,
}


def compute_static_temporal_summary(df: pd.DataFrame, layer: int) -> pd.DataFrame:
    grp = df.groupby(["sl_label", "perturbation_condition", "ablation_target"], as_index=False).agg(
        mean_delta=("delta", "mean"), n=("delta", "count"))
    static = grp[grp.sl_label == "static"].set_index(["perturbation_condition", "ablation_target"])
    temporal = grp[grp.sl_label == "temporal"].set_index(["perturbation_condition", "ablation_target"])
    out = pd.DataFrame({
        "static_delta": static["mean_delta"], "temporal_delta": temporal["mean_delta"],
        "n_clips_static": static["n"], "n_clips_temporal": temporal["n"],
    }).reset_index()
    out["static_temporal_ratio"] = out["static_delta"] / out["temporal_delta"]
    out.insert(0, "layer", layer)
    cols = ["layer", "perturbation_condition", "ablation_target", "static_delta",
            "temporal_delta", "static_temporal_ratio", "n_clips_static", "n_clips_temporal"]
    return out[cols]


def compute_additivity(df: pd.DataFrame, singleton_targets: list[str]) -> pd.DataFrame:
    """sum(5 singletons) - TOP5, per condition — TOP12 excluded: only 5 of its 12
    members have singleton runs, so a sum(singletons) - TOP12 gap would compare
    mismatched feature sets, not a real additivity test (brief only specifies
    this formula for TOP5)."""
    overall = df.groupby(["perturbation_condition", "ablation_target"], as_index=False).agg(
        mean_delta=("delta", "mean"))
    rows = []
    for cond, sub in overall.groupby("perturbation_condition"):
        by_target = sub.set_index("ablation_target")["mean_delta"]
        singleton_sum = float(by_target.loc[singleton_targets].sum())
        top5_delta = float(by_target.loc["TOP5"])
        rows.append({"perturbation_condition": cond, "singleton_sum": singleton_sum,
                     "top5_delta": top5_delta, "additivity_gap": singleton_sum - top5_delta})
    return pd.DataFrame(rows)


def compute_class_detail(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[(df.perturbation_condition == "R") & (df.ablation_target.isin(["TOP5", "TOP12"]))]
    out = sub.groupby(["ablation_target", "class_id", "sl_label"], as_index=False).agg(
        n=("delta", "count"), baseline_logit=("baseline_logit", "mean"),
        ablated_logit=("ablated_logit", "mean"), delta=("delta", "mean"),
        flip_rate=("correct_ablated", lambda s: 1 - s.mean()))
    return out.sort_values(["ablation_target", "delta"], ascending=[True, False])


def main() -> None:
    df = pd.read_parquet(CFG["results_path"])
    singleton_targets = [t for t in df["ablation_target"].unique() if t.startswith("single_")]
    print(f"  {len(df):,} rows  |  conditions: {sorted(df['perturbation_condition'].unique())}")

    summary = compute_static_temporal_summary(df, CFG["layer"])
    additivity = compute_additivity(df, singleton_targets)
    class_detail = compute_class_detail(df)

    out_dir: Path = CFG["out_dir"]
    summary.to_csv(out_dir / "tf_signflip_ablation_summary.csv", index=False)
    additivity.to_csv(out_dir / "tf_signflip_ablation_additivity.csv", index=False)
    class_detail.to_csv(out_dir / "tf_signflip_ablation_class_detail.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(additivity.to_string(index=False))
    print()
    for target in ["TOP5", "TOP12"]:
        sub = class_detail[class_detail.ablation_target == target]
        print(f"-- {target} top 10 by delta --")
        print(sub.head(10).to_string(index=False))
        print(f"-- {target} bottom 10 by delta --")
        print(sub.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
