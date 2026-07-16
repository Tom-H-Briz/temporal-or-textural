"""
L5 clean8 vs L7 clean7 ablation comparison — R + C1, diff-in-diff, additivity.

Reuses L7's existing on-record clean7 result (ablation_results_long_clean7_020726.parquet,
7 singletons + clean7 group) rather than re-running it. L5's clean8 result (8 singletons +
clean8 group) comes from run_ablation.py --layer 5 (see slurm/run_scaffold_ablation_l5.sh).

Both tables carry an explicit caveat on L5 rows only: L5's SAE has a 9.2pp splice-accuracy
hit vs. a much smaller L7 gap (workbook_entry_140726.md #11) — absolute magnitudes aren't
comparable across layers, only deltas relative to each layer's own spliced baseline are.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_l5_vs_l7_comparison.csv   — layer x target x condition
    ablation_l5_vs_l7_additivity.csv   — singleton-sum vs group delta, both layers

Usage:
    uv run python src/stage3_analysis/ablation_l5_vs_l7_comparison.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_targets import SINGLETON_TARGETS as L5_SINGLETONS

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
    "l5_source": ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_clean8_150726.parquet",
    "l7_source": ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_clean7_020726.parquet",
    "out_dir":   ROOT / "outputs/analysis/scaffold_ablation",
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
    """One row per (target, condition): temporal/static logit_drop (mean delta,
    already ablated-vs-spliced-baseline by construction of `delta` upstream),
    diff_in_diff, and flip rates per sl_label."""
    rows = []
    for target in targets:
        for cond in CONDITIONS:
            sub = df[(df["ablation_target"] == target) & (df["perturbation_condition"] == cond)]
            temporal = sub[sub["sl_label"] == "temporal"]
            static   = sub[sub["sl_label"] == "static"]
            rows.append({
                "layer": layer, "target": target, "condition": cond,
                "temporal_logit_drop": temporal["delta"].mean(),
                "static_logit_drop":   static["delta"].mean(),
                "diff_in_diff":        temporal["delta"].mean() - static["delta"].mean(),
                "flip_rate_temporal":  temporal["flip"].mean(),
                "flip_rate_static":    static["flip"].mean(),
                "caveat": L5_CAVEAT if layer == 5 else "",
            })
    return pd.DataFrame(rows)


def build_additivity_table(df: pd.DataFrame, singletons: list[str], group: str, layer: int) -> pd.DataFrame:
    """sum(singleton deltas) vs measured group delta, pooled across sl_label
    ('overall', not split temporal/static) — same convention as
    ablation_summary.py's compute_additivity."""
    rows = []
    for cond in CONDITIONS:
        sub = df[df["perturbation_condition"] == cond]
        singleton_sum = sub[sub["ablation_target"].isin(singletons)].groupby("ablation_target")["delta"].mean().sum()
        group_delta   = sub[sub["ablation_target"] == group]["delta"].mean()
        rows.append({
            "layer": layer, "condition": cond, "group_target": group,
            "singleton_sum": float(singleton_sum), "group_delta": float(group_delta),
            "additivity_gap": float(singleton_sum - group_delta),
            "caveat": L5_CAVEAT if layer == 5 else "",
        })
    return pd.DataFrame(rows)


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

    comparison.to_csv(out_dir / "ablation_l5_vs_l7_comparison.csv", index=False)
    additivity.to_csv(out_dir / "ablation_l5_vs_l7_additivity.csv", index=False)
    print(comparison.drop(columns="caveat").round(4).to_string(index=False))
    print()
    print(additivity.drop(columns="caveat").round(4).to_string(index=False))
    print(f"\n{L5_CAVEAT}")


if __name__ == "__main__":
    main()
