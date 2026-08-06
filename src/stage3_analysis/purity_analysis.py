"""
Increase/decrease purity & crossover at full N — CC brief 06/08.

Replicates the 3-class pilot's purity/mixed finding across the full 33 VM /
31 TF SL-eligible population, plus the magnitude control the pilot couldn't
run. Reads ssv2_{vm,tf}_full_detail.parquet (full_detail_table.py) — no new
SAE/DFA extraction, pure re-aggregation.

Outputs (outputs/analysis/shuffle_reduction_composition/):
    {backbone}_purity_pooled.csv
    {backbone}_purity_crossclass.csv
    {backbone}_purity_magnitude_controlled.csv

Usage:
    uv run python src/stage3_analysis/purity_analysis.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"
BALANCED_RATIO = 0.30  # minor-direction share > this fraction of major-direction -> "balanced"


def purity_pooled(detail: pd.DataFrame) -> pd.DataFrame:
    """1a — pooled-by-class purity, replicates the pilot's method at full N."""
    incdec = detail[detail["bucket"].isin(["increase", "decrease"])]
    counts = incdec.groupby(["class_id", "feature_id", "bucket"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["increase", "decrease"], fill_value=0)
    n_inc, n_dec = counts["increase"], counts["decrease"]
    label = np.where(n_dec == 0, "pure_increase",
             np.where(n_inc == 0, "pure_decrease", "mixed"))
    minor, major = np.minimum(n_inc, n_dec).to_numpy(), np.maximum(n_inc, n_dec).to_numpy()
    balanced = (label == "mixed") & (minor / major > BALANCED_RATIO)
    out = counts.reset_index().rename(columns={"increase": "n_increase", "decrease": "n_decrease"})
    out["label"], out["balanced_flag"] = label, balanced
    return out[["class_id", "feature_id", "label", "n_increase", "n_decrease", "balanced_flag"]]


def purity_crossclass(pooled: pd.DataFrame) -> pd.DataFrame:
    """1b — is a feature's pure/mixed identity consistent across the different
    classes it appears in, not just stable within one class's own content."""
    rows = []
    for fid, grp in pooled.groupby("feature_id"):
        if grp["class_id"].nunique() < 2:
            continue
        labels = set(grp["label"])
        if labels == {"pure_decrease"}:
            flag = "consistent_decrease"
        elif labels == {"pure_increase"}:
            flag = "consistent_increase"
        else:
            flag = "inconsistent"
        per_class = ";".join(f"{c}:{l}" for c, l in zip(grp["class_id"], grp["label"]))
        rows.append({"feature_id": fid, "n_classes_present": grp["class_id"].nunique(),
                     "per_class_labels": per_class, "consistency_flag": flag})
    return pd.DataFrame(rows)


def top_tertile_filter(detail: pd.DataFrame) -> pd.DataFrame:
    """Same well-clear-of-threshold margin already used for sign_flip/noise this
    session: top tertile of |relative_change|, computed once per backbone over
    all increase/decrease instances (not per-class) — same rel_change formula
    top10_detail() uses internally, just not retained in its output."""
    incdec = detail[detail["bucket"].isin(["increase", "decrease"])].copy()
    rel_change = (incdec["signed_shuffle"].abs() - incdec["signed_R"].abs()) / incdec["signed_R"].abs()
    incdec["abs_rel_change"] = rel_change.abs()
    cutoff = incdec["abs_rel_change"].quantile(2 / 3)
    kept = incdec[incdec["abs_rel_change"] >= cutoff]
    return pd.concat([kept, detail[~detail["bucket"].isin(["increase", "decrease"])]], ignore_index=True)


def _mixed_frac(pooled: pd.DataFrame) -> float:
    return float((pooled["label"] == "mixed").mean()) if len(pooled) else float("nan")


def report_raw_vs_controlled(name: str, raw_pooled: pd.DataFrame, ctrl_pooled: pd.DataFrame,
                              raw_cross: pd.DataFrame, ctrl_cross: pd.DataFrame) -> None:
    print(f"\n[{name}] mixed-fraction (pooled, 1a): raw={_mixed_frac(raw_pooled):.3f} "
          f"controlled={_mixed_frac(ctrl_pooled):.3f}")
    for label, df in [("raw", raw_cross), ("controlled", ctrl_cross)]:
        dist = df["consistency_flag"].value_counts(normalize=True) if len(df) else {}
        print(f"  cross-class ({label}, n={len(df)}): {dict(dist)}")


def run_backbone(backbone: str) -> None:
    print(f"Processing {backbone}...")
    detail = pd.read_parquet(OUT_DIR / f"ssv2_{backbone}_full_detail.parquet")
    raw_pooled = purity_pooled(detail)
    raw_cross = purity_crossclass(raw_pooled)
    ctrl_pooled = purity_pooled(top_tertile_filter(detail))
    ctrl_cross = purity_crossclass(ctrl_pooled)

    raw_pooled.to_csv(OUT_DIR / f"{backbone}_purity_pooled.csv", index=False)
    raw_cross.to_csv(OUT_DIR / f"{backbone}_purity_crossclass.csv", index=False)
    ctrl_pooled.to_csv(OUT_DIR / f"{backbone}_purity_magnitude_controlled.csv", index=False)
    report_raw_vs_controlled(backbone, raw_pooled, ctrl_pooled, raw_cross, ctrl_cross)


def main() -> None:
    for backbone in ("vm", "tf"):
        run_backbone(backbone)


if __name__ == "__main__":
    main()
