"""
Clean7 tubelet position analysis — U-shape, half-split, ubiquity.

Task 1: Mean mass per tubelet (0-7), averaged across clean7 features + SL-32 classes.
Task 2: First-half / second-half mass split per class.
Task 3: frac_clips_active per feature (from scaffold_mass_long.parquet, abs_val > 1e-8).

Outputs:
    outputs/analysis/clean7_tubelet_u_shape_080726.csv      (8 rows)
    outputs/analysis/clean7_halfsplit_by_class_080726.csv   (31 rows)
    outputs/analysis/clean7_ubiquity_080726.csv             (updated with source column)
    outputs/analysis/clean7_tubelet_u_shape_080726.png

Usage:
    uv run python src/stage3_analysis/clean7_tubelet_analysis.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

CLEAN7      = [1842, 5578, 1996, 3513, 1990, 3558, 5552]
BAD_CLASSES = {38, 83, 97, 160}
ACTIVE_EPS  = 1e-8

CFG = {
    "tubelet_pq":  ROOT / "outputs/analysis/dfa_per_tubelet_mass/tubelet_position_lock.parquet",
    "mass_long":   ROOT / "outputs/analysis/scaffold_mass_pct/scaffold_mass_long.parquet",
    "acc_csv":     ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "out_dir":     ROOT / "outputs/analysis",
}


def load_filtered(cfg: dict) -> pd.DataFrame:
    acc   = pd.read_csv(cfg["acc_csv"])
    valid = set(acc.loc[acc["accuracy"] >= 0.40, "class_id"])
    df = pd.read_parquet(cfg["tubelet_pq"])
    return df[
        (df["condition"]    == "R") &
        (df["feature_idx"].isin(CLEAN7)) &
        (df["class_id"].isin(valid))
    ].copy()


def task1_u_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Weighted mean mass per tubelet across all clean7 features and classes."""
    df = df.copy()
    df["weighted"] = df["mean_abs"] * df["n_clips"]
    agg = df.groupby("tubelet_idx").agg(
        total_weighted=("weighted", "sum"),
        total_clips   =("n_clips",  "sum"),
    ).reset_index()
    agg["mean_mass"] = agg["total_weighted"] / agg["total_clips"]
    return agg[["tubelet_idx", "mean_mass"]].sort_values("tubelet_idx")


def task2_halfsplit(df: pd.DataFrame) -> pd.DataFrame:
    """First-half (0-3) vs second-half (4-7) mass split per class."""
    df = df.copy()
    df["weighted"] = df["mean_abs"] * df["n_clips"]
    df["half"] = df["tubelet_idx"].apply(lambda t: "first" if t <= 3 else "second")
    agg = df.groupby(["class_id", "sl_label", "half"]).agg(
        mass =("weighted", "sum"),
        clips=("n_clips",  "sum"),
    ).reset_index()
    agg["mean_mass"] = agg["mass"] / agg["clips"]
    pivot = agg.pivot_table(index=["class_id", "sl_label"],
                            columns="half", values="mean_mass").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"first": "first_half_mass", "second": "second_half_mass"})
    pivot["total_mass"]        = pivot["first_half_mass"] + pivot["second_half_mass"]
    pivot["second_half_share"] = pivot["second_half_mass"] / pivot["total_mass"]
    pivot["flagged"]           = ((pivot["second_half_share"] > 0.65) |
                                  (pivot["second_half_share"] < 0.35))
    return pivot.sort_values("second_half_share", ascending=False).reset_index(drop=True)


def task3_ubiquity(cfg: dict) -> pd.DataFrame:
    """frac_clips_active from scaffold_mass_long.parquet (abs_val > 1e-8)."""
    acc   = pd.read_csv(cfg["acc_csv"])
    valid = set(acc.loc[acc["accuracy"] >= 0.40, "class_id"])
    df = pd.read_parquet(cfg["mass_long"])
    df = df[(df["condition"] == "R") &
            (df["feature_idx"].isin(CLEAN7)) &
            (df["class_id"].isin(valid))].copy()
    df["active"] = df["abs_val"] > ACTIVE_EPS
    result = df.groupby("feature_idx").agg(
        n_clips           =("clip_id", "nunique"),
        frac_clips_active_R=("active", "mean"),
    ).reset_index()
    result["source"] = "scaffold_mass_long.parquet abs_val > 1e-8"
    return result


def plot_u_shape(u: pd.DataFrame, out_path: Path) -> None:
    colours = ["#E74C3C" if t == 2 else "#2980B9" for t in u["tubelet_idx"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(u["tubelet_idx"], u["mean_mass"], color=colours, width=0.7)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"T{i}" for i in range(8)], fontsize=10)
    ax.set_xlabel("Tubelet position", fontsize=10)
    ax.set_ylabel("Mean mass (weighted across clean7 + SL-32)", fontsize=9)
    ax.set_title("Clean7 features — mass distribution across tubelet positions (R condition)", fontsize=10)
    ax.annotate("near-absent", xy=(2, u.loc[u["tubelet_idx"]==2, "mean_mass"].values[0]),
                xytext=(2.4, u["mean_mass"].max() * 0.6),
                arrowprops=dict(arrowstyle="->", color="#E74C3C"), color="#E74C3C", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    out = CFG["out_dir"]
    df  = load_filtered(CFG)
    print(f"  {len(df):,} rows  |  {df['class_id'].nunique()} classes  |  {df['feature_idx'].nunique()} features")

    u = task1_u_shape(df)
    u.to_csv(out / "clean7_tubelet_u_shape_080726.csv", index=False)
    print("\nTask 1 — U-shape:")
    print(u.round(5).to_string(index=False))
    plot_u_shape(u, out / "clean7_tubelet_u_shape_080726.png")

    h = task2_halfsplit(df)
    h.to_csv(out / "clean7_halfsplit_by_class_080726.csv", index=False)
    print("\nTask 2 — Half-split (flagged classes >65/35):")
    print(h[h["flagged"]][["class_id","sl_label","first_half_mass","second_half_mass","second_half_share"]].round(4).to_string(index=False))

    ub = task3_ubiquity(CFG)
    ub.to_csv(out / "clean7_ubiquity_080726.csv", index=False)
    print("\nTask 3 — Ubiquity:")
    print(ub.to_string(index=False))


if __name__ == "__main__":
    main()
