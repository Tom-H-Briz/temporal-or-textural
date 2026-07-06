"""
Per-class scaffold feature share breakdown — clean7, R condition, 32 SL classes.

Outputs (outputs/analysis/scaffold_mass_pct/):
    scaffold_class_feature_shares.csv   — wide: one row per class, one col per feature
    scaffold_class_feature_shares_long.csv — long: one row per class × feature

Usage:
    uv run python src/stage3_analysis/scaffold_pct_class_breakdown.py
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

CLEAN7      = [1842, 5578, 1996, 3513, 1990, 3558, 5552]
BAD_CLASSES = {38, 97, 160}
FEAT_ORDER  = [1842, 5552, 3558, 1990, 3513, 5578, 1996]   # pooled rank order

CFG = {
    "source":  ROOT / "outputs/analysis/scaffold_mass_pct/scaffold_pct_per_class_per_feature.csv",
    "out_dir": ROOT / "outputs/analysis/scaffold_mass_pct",
}


def load_filtered(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[
        (df["condition"] == "R") &
        (df["feature_idx"].isin(CLEAN7)) &
        (~df["class_id"].isin(BAD_CLASSES))
    ].copy()


def add_shares(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class_total"] = df.groupby("class_id")["mean_pct"].transform("sum")
    df["share_of_class_pct"] = (df["mean_pct"] / df["class_total"] * 100).round(2)
    df["clean7_total_pct"]   = (df["class_total"] * 100).round(4)
    return df


def build_wide(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=["class_id", "sl_label"],
        columns="feature_idx",
        values="share_of_class_pct",
    )[FEAT_ORDER]
    pivot.columns = [f"f{c}_share_pct" for c in pivot.columns]

    totals = df.groupby(["class_id", "sl_label"])["clean7_total_pct"].first()
    dominant = (
        df.pivot_table(index=["class_id", "sl_label"], columns="feature_idx",
                       values="share_of_class_pct")[FEAT_ORDER]
        .idxmax(axis=1)
        .rename("dominant_feat")
    )
    wide = pivot.join(totals).join(dominant).reset_index()
    return wide.sort_values("clean7_total_pct", ascending=False).reset_index(drop=True)


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["class_id", "sl_label", "feature_idx",
            "mean_pct", "share_of_class_pct", "clean7_total_pct", "n_clips"]
    return df[cols].sort_values(
        ["clean7_total_pct", "share_of_class_pct"], ascending=[False, False]
    ).reset_index(drop=True)


def main() -> None:
    df  = load_filtered(CFG["source"])
    df  = add_shares(df)

    wide = build_wide(df)
    long = build_long(df)

    out = CFG["out_dir"]
    wide.to_csv(out / "scaffold_class_feature_shares.csv", index=False)
    long.to_csv(out / "scaffold_class_feature_shares_long.csv", index=False)

    print(wide.to_string(index=False))
    print(f"\n{len(wide)} classes  |  range: "
          f"{wide['clean7_total_pct'].min():.4f}% – {wide['clean7_total_pct'].max():.4f}%  |  "
          f"std: {wide['clean7_total_pct'].std():.4f}%")
    outliers = wide[wide["f1842_share_pct"] > 40]
    if not outliers.empty:
        print(f"\nConcentration outliers (f1842 > 40% of class total): "
              f"{outliers['class_id'].tolist()}")


if __name__ == "__main__":
    main()
