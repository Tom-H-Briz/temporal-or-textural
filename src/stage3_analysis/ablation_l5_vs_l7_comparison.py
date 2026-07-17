"""
L5 clean8 vs L7 clean7 ablation comparison — R + C1, diff-in-diff, additivity.

Reuses L7's existing on-record clean7 result (ablation_results_long_clean7_020726.parquet,
7 singletons + clean7 group) rather than re-running it. L5's clean8 result (8 singletons +
clean8 group) comes from run_ablation.py --layer 5 (see slurm/run_scaffold_ablation_l5.sh).

Both tables carry an explicit caveat on L5 rows only: L5's SAE has a 9.2pp splice-accuracy
hit vs. a much smaller L7 gap (workbook_entry_140726.md #11) — absolute magnitudes aren't
comparable across layers, only deltas relative to each layer's own spliced baseline are.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_l5_vs_l7_comparison.csv    — layer x target x condition
    ablation_l5_vs_l7_additivity.csv    — singleton-sum vs group delta, both layers
    ablation_l5_vs_l7_class_impact.csv  — per-class flip_rate/delta, group ablation, both layers side by side

Usage:
    uv run python src/stage3_analysis/ablation_l5_vs_l7_comparison.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import load_metadata
from stage3_analysis.ablation_targets import SINGLETON_TARGETS as L5_SINGLETONS

# Same 4-class exclusion used elsewhere (clean7_ubiquity.py, clean7_effect_summary.py) —
# R-accuracy < 40%, too noisy to trust class-level numbers from. Note
# scaffold_pct_class_breakdown.py has a 3-class {38,97,160} variant missing 83;
# this follows the majority (4-class) precedent, not that one.
BAD_CLASSES = {38, 83, 97, 160}

CONDITIONS = ["R", "C1"]
L5_TARGETS = L5_SINGLETONS + ["clean8"]
L7_TARGETS = ["single_1842", "single_5578", "single_1996", "single_3513",
              "single_1990", "single_3558", "single_5552", "clean7"]

L5_CAVEAT = (
    "L5's SAE has a 9.2pp splice-accuracy hit (clip-weighted) vs. a much smaller gap "
    "at L7 (workbook_entry_140726.md #11) — absolute magnitudes are not comparable "
    "across layers. All deltas are ablated-vs-spliced-baseline, relative to each "
    "layer's own unablated SAE-reconstructed model."
)

CFG = {
    "l5_source":  ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_clean8_150726.parquet",
    "l7_source":  ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_clean7_020726.parquet",
    "acc_csv":    ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "val_path":   ROOT / "data/ssv2/labels/validation.json",
    "out_dir":    ROOT / "outputs/analysis/scaffold_ablation",
}


def add_flip_column(df: pd.DataFrame) -> pd.DataFrame:
    """flip = post-ablation, the baseline's rank-1 class no longer beats the
    baseline's rank-2 class. Same definition as ablation_margin_sl.py's
    `flip` column — reused for consistency, not reinvented."""
    base_mat = np.stack(df["baseline_all_logits"].to_numpy()).astype(np.float32)
    abl_mat  = np.stack(df["ablated_all_logits"].to_numpy()).astype(np.float32)
    idx      = np.arange(len(df))
    ranked   = np.argsort(base_mat, axis=1)[:, ::-1]
    r1, r2   = ranked[:, 0], ranked[:, 1]
    margin_abl = abl_mat[idx, r1] - abl_mat[idx, r2]
    out = df.copy()
    out["flip"] = margin_abl < 0
    return out


def build_comparison_table(df: pd.DataFrame, targets: list[str], layer: int) -> pd.DataFrame:
    """One row per (target, condition): temporal/static logit_drop, mean and
    median (median per ablation_median_summary.py's convention — less
    distorted by near-zero/outlier clips), diff_in_diff on both, flip rates."""
    rows = []
    for target in targets:
        for cond in CONDITIONS:
            sub = df[(df["ablation_target"] == target) & (df["perturbation_condition"] == cond)]
            temporal = sub[sub["sl_label"] == "temporal"]
            static   = sub[sub["sl_label"] == "static"]
            rows.append({
                "layer": layer, "target": target, "condition": cond,
                "temporal_logit_drop":        temporal["delta"].mean(),
                "static_logit_drop":          static["delta"].mean(),
                "diff_in_diff":               temporal["delta"].mean() - static["delta"].mean(),
                "temporal_logit_drop_median": temporal["delta"].median(),
                "static_logit_drop_median":   static["delta"].median(),
                "diff_in_diff_median":        temporal["delta"].median() - static["delta"].median(),
                "flip_rate_temporal":         temporal["flip"].mean(),
                "flip_rate_static":           static["flip"].mean(),
                "caveat": L5_CAVEAT if layer == 5 else "",
            })
    return pd.DataFrame(rows)


def build_additivity_table(df: pd.DataFrame, singletons: list[str], group: str, layer: int) -> pd.DataFrame:
    """sum(singleton deltas) vs measured group delta, pooled across sl_label
    ('overall', not split temporal/static) — mean (ablation_summary.py's
    convention) and median (ablation_median_summary.py's convention — median
    is less distorted by near-zero/outlier clips) reported side by side,
    each with its own synergy pct_increase = group / singleton_sum - 1."""
    rows = []
    for cond in CONDITIONS:
        sub = df[df["perturbation_condition"] == cond]
        by_target = sub[sub["ablation_target"].isin(singletons)].groupby("ablation_target")["delta"]
        mean_sum, median_sum = by_target.mean().sum(), by_target.median().sum()
        group_sub = sub[sub["ablation_target"] == group]["delta"]
        mean_group, median_group = group_sub.mean(), group_sub.median()
        rows.append({
            "layer": layer, "condition": cond, "group_target": group,
            "singleton_sum": float(mean_sum), "group_delta": float(mean_group),
            "additivity_gap": float(mean_sum - mean_group),
            "pct_increase": float(mean_group / mean_sum - 1) * 100,
            "singleton_median_sum": float(median_sum), "group_median": float(median_group),
            "median_pct_increase": float(median_group / median_sum - 1) * 100,
            "caveat": L5_CAVEAT if layer == 5 else "",
        })
    return pd.DataFrame(rows)


def build_class_impact(df: pd.DataFrame, group_target: str, layer: int,
                        names: dict, acc_map: dict) -> pd.DataFrame:
    """Per-class impact of the group ablation (R condition only — matches
    the per-class flip-rate breakdown convention already used for L5)."""
    sub = df[(df["ablation_target"] == group_target) & (df["perturbation_condition"] == "R")]
    per_class = sub.groupby(["class_id", "sl_label"], as_index=False).agg(
        n_clips=("flip", "count"), flip_rate=("flip", "mean"),
        mean_delta=("delta", "mean"), median_delta=("delta", "median"),
    )
    per_class["class_name"] = per_class["class_id"].map(names)
    per_class["R_accuracy"] = per_class["class_id"].map(acc_map)
    per_class["flagged_low_acc"] = per_class["class_id"].isin(BAD_CLASSES)
    per_class["layer"] = layer
    return per_class


def build_class_impact_comparison(l5_impact: pd.DataFrame, l7_impact: pd.DataFrame) -> pd.DataFrame:
    """L5 clean8 vs L7 clean7 impact, side by side per class — same 32 SL
    classes at both layers, so class_id/sl_label/class_name align directly."""
    merged = l5_impact.merge(
        l7_impact, on=["class_id", "sl_label", "class_name", "R_accuracy", "flagged_low_acc"],
        suffixes=("_l5", "_l7"),
    )
    merged["flip_rate_diff"] = merged["flip_rate_l5"] - merged["flip_rate_l7"]
    merged["rank_l5"] = merged["flip_rate_l5"].rank(ascending=False, method="min")
    merged["rank_l7"] = merged["flip_rate_l7"].rank(ascending=False, method="min")
    return merged.sort_values("flip_rate_l5", ascending=False).drop(columns=["layer_l5", "layer_l7"])


def main() -> None:
    out_dir: Path = CFG["out_dir"]

    l5 = add_flip_column(pd.read_parquet(CFG["l5_source"]))
    l7 = add_flip_column(pd.read_parquet(CFG["l7_source"]))

    comparison = pd.concat([
        build_comparison_table(l5, L5_TARGETS, layer=5),
        build_comparison_table(l7, L7_TARGETS, layer=7),
    ], ignore_index=True)
    additivity = pd.concat([
        build_additivity_table(l5, L5_SINGLETONS, "clean8", layer=5),
        build_additivity_table(l7, L7_TARGETS[:-1], "clean7", layer=7),
    ], ignore_index=True)

    names   = {v: k for k, v in load_metadata(str(CFG["labels_path"]), str(CFG["val_path"]))[0].items()}
    acc_map = pd.read_csv(CFG["acc_csv"]).set_index("class_id")["accuracy"].to_dict()
    class_impact = build_class_impact_comparison(
        build_class_impact(l5, "clean8", 5, names, acc_map),
        build_class_impact(l7, "clean7", 7, names, acc_map),
    )

    comparison.to_csv(out_dir / "ablation_l5_vs_l7_comparison.csv", index=False)
    additivity.to_csv(out_dir / "ablation_l5_vs_l7_additivity.csv", index=False)
    class_impact.to_csv(out_dir / "ablation_l5_vs_l7_class_impact.csv", index=False)
    print(comparison.drop(columns="caveat").round(4).to_string(index=False))
    print()
    print(additivity.drop(columns="caveat").round(4).to_string(index=False))
    print()
    print(class_impact.round(4).to_string(index=False))
    print(f"\n{L5_CAVEAT}")


if __name__ == "__main__":
    main()
