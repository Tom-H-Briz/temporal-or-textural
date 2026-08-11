"""
Per-class DFA mass share, logit damage, and flip rate — TF sign-flip TOP5/TOP12.

DFA mass fraction is condition-independent (always from signed_vec_R — the
real-condition attribution, same convention used for the earlier population-
level mass/enrichment numbers), computed once per (class, target). Logit
damage and flip rate come from the ablation results and are reported for
both R and C side by side, since those genuinely differ by condition.

Outputs:
    outputs/analysis/scaffold_ablation/tf_signflip_class_summary.csv

Usage:
    uv run python src/stage3_analysis/tf_signflip_class_summary.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

DFA_PARQUET = ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet"
ABLATION_PARQUET = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_results_long.parquet"
TF_ACC_CSV = ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv"
OUT_PATH = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_class_summary.csv"

TARGETS = {
    "TOP5":  [3029, 1517, 2090, 2057, 2156],
    "TOP12": [3029, 1517, 2090, 2057, 2156, 1588, 4590, 3813, 6029, 622, 1371, 4134],
}


def dfa_mass_share_per_class() -> pd.DataFrame:
    """Per clip, fraction of total |DFA| mass (signed_vec_R) each target's
    features explain — then averaged per class. Target-independent of
    perturbation condition, unlike logit damage and flip rate below."""
    df = pd.read_parquet(DFA_PARQUET, columns=["clip_id", "class_id", "signed_vec_R", "total_abs_R"])
    mat = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
    rows = []
    for target, indices in TARGETS.items():
        frac = np.abs(mat[:, indices]).sum(axis=1) / df["total_abs_R"].to_numpy()
        rows.append(pd.DataFrame({"class_id": df["class_id"], "ablation_target": target,
                                  "dfa_mass_fraction": frac}))
    per_clip = pd.concat(rows, ignore_index=True)
    return per_clip.groupby(["class_id", "ablation_target"])["dfa_mass_fraction"].mean().reset_index()


def logit_damage_and_flip_rate() -> pd.DataFrame:
    df = pd.read_parquet(ABLATION_PARQUET)
    df = df[df["ablation_target"].isin(TARGETS.keys())]
    agg = df.groupby(["class_id", "ablation_target", "perturbation_condition"]).agg(
        n_clips=("clip_id", "count"), mean_logit_damage=("delta", "mean"),
        flip_rate=("correct_ablated", lambda s: 1 - s.mean()),
    ).reset_index()
    wide = agg.pivot(index=["class_id", "ablation_target"],
                     columns="perturbation_condition", values=["n_clips", "mean_logit_damage", "flip_rate"])
    wide.columns = [f"{metric}_{cond}" for metric, cond in wide.columns]
    return wide.reset_index()


def main() -> None:
    mass = dfa_mass_share_per_class()
    outcomes = logit_damage_and_flip_rate()
    templates = pd.read_csv(TF_ACC_CSV, usecols=["class_id", "template"])

    out = mass.merge(outcomes, on=["class_id", "ablation_target"]).merge(templates, on="class_id")
    cols = ["class_id", "template", "ablation_target", "dfa_mass_fraction",
            "n_clips_R", "mean_logit_damage_R", "flip_rate_R",
            "n_clips_C", "mean_logit_damage_C", "flip_rate_C"]
    out = out[cols].sort_values(["ablation_target", "flip_rate_R"], ascending=[True, False])
    out.to_csv(OUT_PATH, index=False)
    print(f"{len(out)} rows -> {OUT_PATH}")
    for target in TARGETS:
        sub = out[out.ablation_target == target]
        print(f"\n=== {target} ===")
        print(sub.drop(columns="ablation_target").to_string(index=False))


if __name__ == "__main__":
    main()
