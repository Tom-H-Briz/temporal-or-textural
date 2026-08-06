"""
Within-class vs. between-class decomposition — CC brief 06/08 addendum, item 4 redo.

Original scoping (superseded) used shuffle_reduction_composition.py — a
deprecated three-way (decrease/same/increase) script that pools sign_flip
into decrease/increase rather than splitting it out, and aggregates by
mean_abs_R per class rather than per-clip. Re-verified directly from
ssv2_{vm,tf}_full_detail.parquet (same source as items 1/2/3/5, via
class_feature_breakdown.py's existing per_clip_fracs — not reimplemented):
class-level frac_X vs survival-rate correlation, four-way split —

    VM: frac_sign_flip r=-0.721 p<0.0001, frac_decrease r=+0.686 p<0.0001, frac_increase r=+0.020 p=0.91
    TF: frac_sign_flip r=-0.643 p=0.0001,  frac_decrease r=-0.250 p=0.18,   frac_increase r=-0.141 p=0.45

frac_increase is not significant in either backbone once sign_flip is
correctly split out — confirms the brief's correction. Targets here:
frac_decrease + frac_sign_flip (VM), frac_sign_flip only (TF).

Outputs (outputs/analysis/shuffle_reduction_composition/):
    {backbone}_within_class_correct_incorrect.csv

Usage:
    uv run python src/stage3_analysis/purity_within_class.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.class_feature_breakdown import per_clip_fracs

OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"
TARGETS = {"vm": ["frac_decrease", "frac_sign_flip"], "tf": ["frac_sign_flip"]}


def class_frac_table(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-clip frac_X (per_clip_fracs, reused unchanged) plus class_id — the
    same per-clip aggregation class_feature_breakdown.py already uses, just
    joined back to class_id since per_clip_fracs only keys on clip_id."""
    fracs = per_clip_fracs(detail)
    class_map = detail.drop_duplicates("clip_id").set_index("clip_id")["class_id"]
    fracs["class_id"] = fracs["clip_id"].map(class_map)
    return fracs


def between_class_correlation(fracs: pd.DataFrame, target: str) -> tuple[float, float]:
    by_class = fracs.groupby("class_id").agg(mean_frac=(target, "mean"),
                                              survival=("correct_under_shuffle", "mean"))
    r = stats.pearsonr(by_class["mean_frac"], by_class["survival"])
    return float(r.statistic), float(r.pvalue)


def within_class_split(fracs: pd.DataFrame, target: str) -> pd.DataFrame:
    """Per class: mean_frac_X among clips that survived shuffle vs clips that
    didn't — the individual-clip-level test §7 ran for sign_flip, generalised
    to whichever frac_X the between-class correlation actually singles out."""
    rows = []
    for cid, grp in fracs.groupby("class_id"):
        correct, incorrect = grp[grp["correct_under_shuffle"]], grp[~grp["correct_under_shuffle"]]
        rows.append({
            "class_id": cid, "target": target,
            "n_correct": len(correct), "n_incorrect": len(incorrect),
            "mean_frac_correct": correct[target].mean() if len(correct) else float("nan"),
            "mean_frac_incorrect": incorrect[target].mean() if len(incorrect) else float("nan"),
            "class_survival_rate": grp["correct_under_shuffle"].mean(),
        })
    out = pd.DataFrame(rows)
    out["within_class_gap"] = out["mean_frac_incorrect"] - out["mean_frac_correct"]
    return out


def report_decomposition(name: str, target: str, between_r: float, between_p: float,
                          split_df: pd.DataFrame) -> None:
    valid = split_df.dropna(subset=["within_class_gap"])
    expected_sign = -np.sign(between_r)  # higher frac hurts survival (r<0) -> incorrect clips should show higher frac
    consistent = (np.sign(valid["within_class_gap"]) == expected_sign).mean() if len(valid) else float("nan")
    print(f"  [{name}] {target}: between-class r={between_r:+.3f} (p={between_p:.4f})  |  "
          f"within-class mean_gap={valid['within_class_gap'].mean():+.4f}  "
          f"{consistent:.1%} of classes match the between-class direction (n={len(valid)})")


def main() -> None:
    for backbone, targets in TARGETS.items():
        print(f"Processing {backbone}...")
        detail = pd.read_parquet(OUT_DIR / f"ssv2_{backbone}_full_detail.parquet")
        fracs = class_frac_table(detail)
        rows = []
        for target in targets:
            r, p = between_class_correlation(fracs, target)
            split_df = within_class_split(fracs, target)
            report_decomposition(backbone, target, r, p, split_df)
            split_df["between_class_r"], split_df["between_class_p"] = r, p
            rows.append(split_df)
        pd.concat(rows, ignore_index=True).to_csv(
            OUT_DIR / f"{backbone}_within_class_correct_incorrect.csv", index=False)


if __name__ == "__main__":
    main()
