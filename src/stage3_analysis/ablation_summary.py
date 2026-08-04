"""
Ablation summary — aggregates ablation_results_long.parquet.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_summary.csv         — mean_delta per (sl_group, condition, target)
    ablation_diff_in_diff.csv    — temporal − static delta per (condition, target)
    ablation_additivity.csv      — sum(singleton deltas) − clean7 delta, per condition
    ablation_dfa_agreement.csv   — DFA sign vs ablation delta direction match + Spearman r

Usage:
    uv run python src/stage3_analysis/ablation_summary.py
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_targets import get_targets, group_targets, singleton_targets
from stage3_analysis.scaffold_selection_consolidated import _resolve_mass_delta_parquet

CFG = {
    "out_dir": ROOT / "outputs/analysis/scaffold_ablation",
}


def _parse_run_tag(path: Path) -> tuple[str, int, int]:
    """Extract (dataset, layer, sae_k) from run_ablation.py's filename convention:
    ablation_results_long_{dataset}_l{layer}_job{job_label}_k{sae_k}.parquet —
    so the matching mass-delta parquet (same dataset, layer, SAE) can be
    resolved, not whatever unrelated file happens to be hardcoded. Dataset
    token is optional in the regex — pre-03/08 SSv2 runs have no token in their
    filename, and defaulting those to "ssv2" keeps existing files resolving."""
    m = re.match(r"ablation_results_long_(?:(ssv2|kinetics400)_)?l(\d+)_job.+_k(\d+)\.parquet$", path.name)
    if not m:
        raise ValueError(f"Can't parse dataset/layer/sae_k from filename: {path.name}")
    return m.group(1) or "ssv2", int(m.group(2)), int(m.group(3))

CONDITIONS = ["R", "C1"]


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["sl_label", "perturbation_condition", "ablation_target"]
    by_sl = df.groupby(grp_cols, as_index=False).agg(
        mean_delta=("delta", "mean"), n_clips=("delta", "count")
    )
    overall = df.groupby(["perturbation_condition", "ablation_target"], as_index=False).agg(
        mean_delta=("delta", "mean"), n_clips=("delta", "count")
    )
    overall["sl_label"] = "overall"
    cols = ["sl_label", "perturbation_condition", "ablation_target", "mean_delta", "n_clips"]
    return pd.concat([by_sl[cols], overall[cols]], ignore_index=True)


def compute_diff_in_diff(summary: pd.DataFrame) -> pd.DataFrame:
    idx = ["perturbation_condition", "ablation_target"]
    temporal = summary[summary["sl_label"] == "temporal"].set_index(idx)["mean_delta"]
    static   = summary[summary["sl_label"] == "static"].set_index(idx)["mean_delta"]
    return (temporal - static).reset_index().rename(columns={"mean_delta": "diff_in_diff"})


def compute_additivity(summary: pd.DataFrame, singleton: list[str], group_names: list[str]) -> pd.DataFrame:
    overall = summary[summary["sl_label"] == "overall"]
    rows = []
    for cond in CONDITIONS:
        sub = overall[overall["perturbation_condition"] == cond].set_index("ablation_target")["mean_delta"]
        singleton_sum = float(sub.loc[singleton].sum())
        for group in group_names:
            group_delta = float(sub.loc[group])
            rows.append({
                "perturbation_condition": cond,
                "group_target":           group,
                "singleton_sum":          singleton_sum,
                "group_delta":            group_delta,
                "additivity_gap":         singleton_sum - group_delta,
            })
    return pd.DataFrame(rows)


def compute_dfa_signed_sums(mass_delta_path: Path, targets: dict[str, list[int]]) -> pd.DataFrame:
    """Long-format dfa_signed_sum per (clip_id, condition, target) from the
    mass-delta parquet matching this run's layer/SAE (resolved by the caller
    via _resolve_mass_delta_parquet, not a filename glob)."""
    src = pd.read_parquet(mass_delta_path, columns=["clip_id", "signed_vec_R", "signed_vec_C1"])
    chunks = []
    for cond in CONDITIONS:
        mat = np.stack(src[f"signed_vec_{cond}"].to_numpy()).astype(np.float32)  # (N, 6144)
        for target_name, indices in targets.items():
            chunk = src[["clip_id"]].copy()
            chunk["perturbation_condition"] = cond
            chunk["ablation_target"]        = target_name
            chunk["dfa_signed_sum"]         = mat[:, indices].sum(axis=1)
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def compute_dfa_agreement(ablation_df: pd.DataFrame, dfa_sums: pd.DataFrame) -> pd.DataFrame:
    """Direction match rate and Spearman r between dfa_signed_sum and ablation delta."""
    merged = ablation_df[["clip_id", "perturbation_condition", "ablation_target", "delta"]].merge(
        dfa_sums, on=["clip_id", "perturbation_condition", "ablation_target"]
    )
    merged["direction_match"] = np.sign(merged["dfa_signed_sum"]) == np.sign(merged["delta"])
    rows = []
    for (target, cond), grp in merged.groupby(["ablation_target", "perturbation_condition"]):
        rho = grp[["dfa_signed_sum", "delta"]].corr(method="spearman").iloc[0, 1]
        rows.append({
            "ablation_target":        target,
            "perturbation_condition": cond,
            "direction_match_rate":   round(float(grp["direction_match"].mean()), 4),
            "rank_correlation":       round(float(rho), 4),
            "n_clips":                len(grp),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_parquet", type=Path,
                        help="path to run_ablation.py's ablation_results_long_l{layer}_job{job}_k{k}.parquet")
    args = parser.parse_args()
    dataset, layer, sae_k = _parse_run_tag(args.results_parquet)
    suffix = args.results_parquet.stem.removeprefix("ablation_results_long_")

    targets = get_targets(dataset, layer)
    singleton, group_names = singleton_targets(targets), group_targets(targets)

    df = pd.read_parquet(args.results_parquet)
    print(f"  {len(df):,} rows  |  conditions: {sorted(df['perturbation_condition'].unique())}")

    out_dir: Path = CFG["out_dir"]
    summary = compute_summary(df)
    did     = compute_diff_in_diff(summary)
    add     = compute_additivity(summary, singleton, group_names)

    mass_delta_path = _resolve_mass_delta_parquet(
        {"name": suffix, "layer": layer, "sae_k": sae_k, "dataset": dataset})
    dfa_sums = compute_dfa_signed_sums(mass_delta_path, targets)
    agree    = compute_dfa_agreement(df, dfa_sums)

    summary.to_csv(out_dir / f"ablation_summary_{suffix}.csv", index=False)
    did.to_csv(out_dir / f"ablation_diff_in_diff_{suffix}.csv", index=False)
    add.to_csv(out_dir / f"ablation_additivity_{suffix}.csv", index=False)
    agree.to_csv(out_dir / f"ablation_dfa_agreement_{suffix}.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(did.to_string(index=False))
    print()
    print(add.to_string(index=False))
    print()
    print(agree.to_string(index=False))


if __name__ == "__main__":
    main()
