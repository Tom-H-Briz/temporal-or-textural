"""
Clean7 feature ubiquity check — frac_clips_active per feature per SL class.

Source: scaffold_mass_long.parquet (abs_val already derived, no re-extraction).
Activity threshold: abs_val > 1e-8 (matches per_class_feature_delta.py convention).

Output: outputs/analysis/clean7_ubiquity_080726.csv

Usage:
    uv run python src/stage3_analysis/clean7_ubiquity.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

CLEAN7      = [1842, 5578, 1996, 3513, 1990, 3558, 5552]
BAD_CLASSES = {38, 83, 97, 160}
ACTIVE_EPS  = 1e-8

CFG = {
    "source":   ROOT / "outputs/analysis/scaffold_mass_pct/scaffold_mass_long.parquet",
    "acc_csv":  ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "out":      ROOT / "outputs/analysis/clean7_ubiquity_080726.csv",
}


def main() -> None:
    acc   = pd.read_csv(CFG["acc_csv"])
    valid = set(acc.loc[acc["accuracy"] >= 0.40, "class_id"])

    df = pd.read_parquet(CFG["source"])
    df = df[
        (df["condition"]    == "R") &
        (df["feature_idx"].isin(CLEAN7)) &
        (df["class_id"].isin(valid))
    ].copy()
    df["active"] = df["abs_val"] > ACTIVE_EPS
    print(f"  {len(df):,} rows  |  {df['clip_id'].nunique():,} clips  |  {df['class_id'].nunique()} classes")

    # Per feature overall
    overall = df.groupby("feature_idx").agg(
        overall_n_clips  =("clip_id", "nunique"),
        overall_frac_active=("active", "mean"),
    ).reset_index()

    # Per feature × class
    per_class = df.groupby(["feature_idx", "class_id", "sl_label"]).agg(
        n_clips    =("clip_id", "count"),
        frac_active=("active", "mean"),
    ).reset_index()

    out = per_class.merge(overall, on="feature_idx")
    out = out.sort_values(["feature_idx", "frac_active"]).reset_index(drop=True)

    out.to_csv(CFG["out"], index=False)
    print(f"  → {CFG['out']}")
    print()
    print(overall.round(6).to_string(index=False))
    print()
    low = out[out["frac_active"] < 1.0]
    if low.empty:
        print("All features active on 100% of clips in every class.")
    else:
        print(f"Classes with frac_active < 1.0:\n{low.to_string(index=False)}")


if __name__ == "__main__":
    main()
