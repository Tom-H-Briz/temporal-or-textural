"""
Ablation summary — clean7 run (7 scaffold singletons + iso4061 + clean7 + all8).

Reads ablation_results_long_clean7_020726.parquet.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_summary_clean7.csv
    ablation_diff_in_diff_clean7.csv
    ablation_additivity_clean7.csv
    ablation_dfa_agreement_clean7.csv

Usage:
    uv run python src/stage3_analysis/ablation_summary_7.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

SINGLETON_TARGETS = ["single_1842", "single_5578", "single_1996", "single_3513",
                     "single_1990", "single_3558", "single_5552"]
GROUP_TARGETS     = ["clean7", "all8"]
CONDITIONS        = ["R", "C1"]

CFG = {
    "source":   ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_clean7_020726.parquet",
    "src_glob": "outputs/analysis/**/dfa_mass_delta_vm_c1.parquet",
    "out_dir":  ROOT / "outputs/analysis/scaffold_ablation",
}


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = ["sl_label", "perturbation_condition", "ablation_target"]
    by_sl = df.groupby(grp_cols, as_index=False).agg(
        mean_delta=("delta", "mean"), median_delta=("delta", "median"), n_clips=("delta", "count")
    )
    overall = df.groupby(["perturbation_condition", "ablation_target"], as_index=False).agg(
        mean_delta=("delta", "mean"), median_delta=("delta", "median"), n_clips=("delta", "count")
    )
    overall["sl_label"] = "overall"
    cols = ["sl_label", "perturbation_condition", "ablation_target", "mean_delta", "median_delta", "n_clips"]
    return pd.concat([by_sl[cols], overall[cols]], ignore_index=True)


def compute_diff_in_diff(summary: pd.DataFrame) -> pd.DataFrame:
    idx = ["perturbation_condition", "ablation_target"]
    temporal = summary[summary["sl_label"] == "temporal"].set_index(idx)["mean_delta"]
    static   = summary[summary["sl_label"] == "static"].set_index(idx)["mean_delta"]
    return (temporal - static).reset_index().rename(columns={"mean_delta": "diff_in_diff"})


def compute_additivity(summary: pd.DataFrame) -> pd.DataFrame:
    overall = summary[summary["sl_label"] == "overall"]
    rows = []
    for cond in CONDITIONS:
        sub = overall[overall["perturbation_condition"] == cond].set_index("ablation_target")["mean_delta"]
        singleton_sum = float(sub.loc[SINGLETON_TARGETS].sum())
        for grp in GROUP_TARGETS:
            group_delta = float(sub.loc[grp])
            rows.append({"perturbation_condition": cond, "group_target": grp,
                         "singleton_sum": singleton_sum, "group_delta": group_delta,
                         "additivity_gap": singleton_sum - group_delta})
    return pd.DataFrame(rows)


def compute_dfa_agreement(df: pd.DataFrame) -> pd.DataFrame:
    matches = list(ROOT.glob(CFG["src_glob"]))
    assert len(matches) == 1
    src = pd.read_parquet(matches[0], columns=["clip_id", "signed_vec_R"])
    all_targets = SINGLETON_TARGETS + GROUP_TARGETS + ["iso4061"]
    feat_map = {
        "single_1842": [1842], "single_5578": [5578], "single_1996": [1996],
        "single_3513": [3513], "single_1990": [1990], "single_3558": [3558],
        "single_5552": [5552], "iso4061": [4061],
        "clean7": [1842, 5578, 1996, 3513, 1990, 3558, 5552],
        "all8":   [1842, 5578, 1996, 3513, 1990, 3558, 4061, 5552],
    }
    mat = np.stack(src["signed_vec_R"].to_numpy()).astype(np.float32)
    chunks = []
    for cond in CONDITIONS:
        for tname, idxs in feat_map.items():
            chunk = src[["clip_id"]].copy()
            chunk["perturbation_condition"] = cond
            chunk["ablation_target"]        = tname
            chunk["dfa_signed_sum"]         = mat[:, idxs].sum(axis=1)
            chunks.append(chunk)
    dfa = pd.concat(chunks, ignore_index=True)
    merged = df[["clip_id","perturbation_condition","ablation_target","delta"]].merge(
        dfa, on=["clip_id","perturbation_condition","ablation_target"]
    )
    merged["direction_match"] = np.sign(merged["dfa_signed_sum"]) == np.sign(merged["delta"])
    rows = []
    for (target, cond), grp in merged.groupby(["ablation_target","perturbation_condition"]):
        rho = grp[["dfa_signed_sum","delta"]].corr(method="spearman").iloc[0,1]
        rows.append({"ablation_target": target, "perturbation_condition": cond,
                     "direction_match_rate": round(float(grp["direction_match"].mean()),4),
                     "rank_correlation": round(float(rho),4), "n_clips": len(grp)})
    return pd.DataFrame(rows)


def load_labels() -> dict:
    with open(ROOT / "data/ssv2/labels/validation.json") as f:
        return {str(c["id"]): c["label"] for c in json.load(f)}


def compute_class_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Per class × target aggregate — R condition, singletons + clean7 only."""
    targets = SINGLETON_TARGETS + ["clean7"]
    sub = df[(df["perturbation_condition"] == "R") & (df["ablation_target"].isin(targets))]
    return sub.groupby(["class_id", "sl_label", "ablation_target"], as_index=False).agg(
        n_clips     =("delta", "count"),
        mean_delta  =("delta", "mean"),
        median_delta=("delta", "median"),
    ).sort_values(["ablation_target", "median_delta"], ascending=[True, False]).reset_index(drop=True)


def compute_class_examples(df: pd.DataFrame, label_map: dict, n: int = 3) -> pd.DataFrame:
    """Top n most and least affected clips per class × target, R condition."""
    targets = SINGLETON_TARGETS + ["clean7"]
    sub = df[(df["perturbation_condition"] == "R") & (df["ablation_target"].isin(targets))].copy()
    sub["label"] = sub["clip_id"].map(label_map)
    rows = []
    for (cls, sl, tgt), grp in sub.groupby(["class_id", "sl_label", "ablation_target"]):
        for role, clips in [("most_affected", grp.nlargest(n, "delta")),
                            ("least_affected", grp.nsmallest(n, "delta"))]:
            for _, row in clips.iterrows():
                rows.append({"class_id": cls, "sl_label": sl, "ablation_target": tgt,
                             "role": role, "clip_id": row["clip_id"],
                             "delta": row["delta"], "label": row["label"]})
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(CFG["source"])
    print(f"  {len(df):,} rows  |  conditions: {sorted(df['perturbation_condition'].unique())}")

    out = CFG["out_dir"]
    summary = compute_summary(df)
    did     = compute_diff_in_diff(summary)
    add     = compute_additivity(summary)
    agree   = compute_dfa_agreement(df)

    labels  = load_labels()
    impact  = compute_class_impact(df)
    examples = compute_class_examples(df, labels)

    summary.to_csv(out / "ablation_summary_clean7.csv", index=False)
    did.to_csv(out / "ablation_diff_in_diff_clean7.csv", index=False)
    add.to_csv(out / "ablation_additivity_clean7.csv", index=False)
    agree.to_csv(out / "ablation_dfa_agreement_clean7.csv", index=False)
    impact.to_csv(out / "ablation_class_impact_clean7.csv", index=False)
    examples.to_csv(out / "ablation_class_examples_clean7.csv", index=False)
    print(f"  class impact → ablation_class_impact_clean7.csv ({len(impact)} rows)")
    print(f"  clip examples → ablation_class_examples_clean7.csv ({len(examples)} rows)")

    print(summary[summary["sl_label"]=="overall"].to_string(index=False))
    print()
    print(did.to_string(index=False))
    print()
    print(add.to_string(index=False))


if __name__ == "__main__":
    main()
