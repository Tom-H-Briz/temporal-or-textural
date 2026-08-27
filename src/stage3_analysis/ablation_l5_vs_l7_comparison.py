"""
L5 vs L7 ablation comparison — R + C1, diff-in-diff, additivity. Post-fix
(job7ep) rebuild, 24/08 — the prior clean7/clean8 version predates the 30/07
bias fix (its own caveat cited the since-retracted 9.2pp splice-accuracy
figure) and its SINGLETON_TARGETS import no longer even exists in
ablation_targets.py. Targets now resolved live via get_targets()/
group_targets() — same accumulating registry ablation_cross_l5_l7.py uses —
instead of hardcoded pre-fix feature-index lists.

L7's group target ("all4") is 3 gate-passed scaffold members (5165/6021/6032)
plus one deliberately-included near-miss (3347) — see ablation_targets.py's
own registry comment. Not the pure 3-member scaffold; flagged, not silently
conflated (same caveat as sign_flip_results.py's VM_L7 row).

Additivity table now also reports the group's prediction flip rate and
post-ablation accuracy (24/08, Tom), not just the logit-delta additivity
gap. flip_rate here is 1 - correct_ablated.mean() (matches sign_flip_
results.py's convention) — deliberately NOT this file's own "flip" column
below (baseline-rank1-vs-rank2 margin crossing zero), which answers a
different question (does ablation overturn the model's own prior top pick,
not whether it's right against ground truth).

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_l5_vs_l7_comparison_job7ep.csv    — layer x target x condition
    ablation_l5_vs_l7_additivity_job7ep.csv    — singleton-sum vs group delta
                                                  + group flip_rate/accuracy
    ablation_l5_vs_l7_class_impact_job7ep.csv  — per-class flip_rate/delta,
                                                  group ablation, both layers

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
from stage3_analysis.ablation_targets import get_targets, group_targets

# Same 4-class exclusion used elsewhere (clean7_ubiquity.py, clean7_effect_summary.py) —
# R-accuracy < 40%, too noisy to trust class-level numbers from. Note
# scaffold_pct_class_breakdown.py has a 3-class {38,97,160} variant missing 83;
# this follows the majority (4-class) precedent, not that one.
BAD_CLASSES = {38, 83, 97, 160}

CONDITIONS = ["R", "C1"]
L5_ALL_TARGETS = get_targets("ssv2", 5)
L7_ALL_TARGETS = get_targets("ssv2", 7)
L5_GROUP, L7_GROUP = group_targets(L5_ALL_TARGETS)[0], group_targets(L7_ALL_TARGETS)[0]
L5_SINGLETONS = [t for t in L5_ALL_TARGETS if t != L5_GROUP]
L7_SINGLETONS = [t for t in L7_ALL_TARGETS if t != L7_GROUP]
L5_TARGETS, L7_TARGETS = L5_SINGLETONS + [L5_GROUP], L7_SINGLETONS + [L7_GROUP]

CFG = {
    "l5_source":  ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_job7ep_k64.parquet",
    "l7_source":  ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l7_job7ep_k64.parquet",
    "acc_csv":    ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "val_path":   ROOT / "data/ssv2/labels/validation.json",
    "out_dir":    ROOT / "outputs/analysis/scaffold_ablation",
}


def add_flip_column(df: pd.DataFrame) -> pd.DataFrame:
    """flip = ablated prediction no longer matches the true class
    (~correct_ablated). Changed 24/08 (Tom) from the original rank1-vs-rank2
    baseline-margin definition (does ablation overturn the model's own prior
    top pick specifically against its own runner-up) — that undercounts real
    flips where a third-ranked class overtakes rank-1 instead of the
    runner-up, and disagreed with build_additivity_table's group_flip_rate
    by a wide margin (9.3% vs 15.5% on SSv2 L5/all7/R) for exactly that
    reason. Now matches sign_flip_results.py's convention throughout this
    project. No longer the same definition as ablation_margin_sl.py's `flip`
    column — that file is unchanged, deliberately out of scope here."""
    out = df.copy()
    out["flip"] = ~out["correct_ablated"]
    return out


def build_comparison_table(df: pd.DataFrame, targets: list[str], layer: int) -> pd.DataFrame:
    """One row per (target, condition): temporal/static logit_drop, mean and
    median (median per ablation_median_summary.py's convention — less
    distorted by near-zero/outlier clips), diff_in_diff on both, flip rates.

    flip_rate_overall is computed directly on the full (unsplit) subset, not
    combined from flip_rate_temporal/flip_rate_static after the fact —
    temporal and static are different-sized groups, so summing or averaging
    the two subgroup rates naively does not equal the true combined rate
    (confirmed 24/08: naive sum overstated it ~2x on SSv2 L5/all7/R)."""
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
                "flip_rate_overall":          sub["flip"].mean(),
                "n_temporal":                 len(temporal),
                "n_static":                   len(static),
            })
    return pd.DataFrame(rows)


def build_additivity_table(df: pd.DataFrame, singletons: list[str], group: str, layer: int) -> pd.DataFrame:
    """sum(singleton deltas) vs measured group delta, pooled across sl_label
    ('overall', not split temporal/static) — mean (ablation_summary.py's
    convention) and median (ablation_median_summary.py's convention — median
    is less distorted by near-zero/outlier clips) reported side by side,
    each with its own synergy pct_increase = group / singleton_sum - 1.

    Also reports the group's own prediction flip rate and resulting
    post-ablation accuracy (24/08, Tom) — logit-delta additivity alone
    doesn't say how many predictions actually changed or what accuracy looks
    like with the group ablated. baseline_accuracy is computed from this
    study's own stored baseline_all_logits (argmax vs class_id), matching
    sign_flip_results.py's VM convention — self-contained, no external join."""
    rows = []
    for cond in CONDITIONS:
        sub = df[df["perturbation_condition"] == cond]
        by_target = sub[sub["ablation_target"].isin(singletons)].groupby("ablation_target")["delta"]
        mean_sum, median_sum = by_target.mean().sum(), by_target.median().sum()
        group_sub = sub[sub["ablation_target"] == group]
        mean_group, median_group = group_sub["delta"].mean(), group_sub["delta"].median()
        baseline_correct = group_sub.apply(
            lambda r: np.argmax(r["baseline_all_logits"]) == r["class_id"], axis=1)
        ablated_accuracy = group_sub["correct_ablated"].mean()
        rows.append({
            "layer": layer, "condition": cond, "group_target": group,
            "singleton_sum": float(mean_sum), "group_delta": float(mean_group),
            "additivity_gap": float(mean_sum - mean_group),
            "pct_increase": float(mean_group / mean_sum - 1) * 100,
            "singleton_median_sum": float(median_sum), "group_median": float(median_group),
            "median_pct_increase": float(median_group / median_sum - 1) * 100,
            "n_clips": len(group_sub),
            "group_flip_rate": float(1 - ablated_accuracy),
            "group_baseline_accuracy": float(baseline_correct.mean()),
            "group_ablated_accuracy": float(ablated_accuracy),
            "group_accuracy_drop": float(baseline_correct.mean() - ablated_accuracy),
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
        build_additivity_table(l5, L5_SINGLETONS, L5_GROUP, layer=5),
        build_additivity_table(l7, L7_SINGLETONS, L7_GROUP, layer=7),
    ], ignore_index=True)

    names   = {v: k for k, v in load_metadata(str(CFG["labels_path"]), str(CFG["val_path"]))[0].items()}
    acc_map = pd.read_csv(CFG["acc_csv"]).set_index("class_id")["accuracy"].to_dict()
    class_impact = build_class_impact_comparison(
        build_class_impact(l5, L5_GROUP, 5, names, acc_map),
        build_class_impact(l7, L7_GROUP, 7, names, acc_map),
    )

    comparison.to_csv(out_dir / "ablation_l5_vs_l7_comparison_job7ep.csv", index=False)
    additivity.to_csv(out_dir / "ablation_l5_vs_l7_additivity_job7ep.csv", index=False)
    class_impact.to_csv(out_dir / "ablation_l5_vs_l7_class_impact_job7ep.csv", index=False)
    print(comparison.round(4).to_string(index=False))
    print()
    print(additivity.round(4).to_string(index=False))
    print()
    print(class_impact.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
