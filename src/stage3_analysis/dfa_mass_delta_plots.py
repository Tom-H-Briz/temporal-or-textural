"""
DFA mass delta visualisation — KDE and scatter plots.

Reads dfa_mass_delta.parquet + per-class accuracy CSVs.
No DFA engine calls.

Outputs:
    outputs/analysis/dfa_mass_delta/kde_delta_temporal_vs_static.png
    outputs/analysis/dfa_mass_delta/scatter_delta_vs_accdrop.png
    outputs/analysis/dfa_mass_delta/per_class_summary.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

SL_COLOURS = {"temporal": "steelblue", "static": "darkorange"}

DFA_CLASSES = [0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40, 41,
               42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164, 168, 169, 171, 173]

PATHS = {
    "parquet":    str(ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet"),
    "acc_R":      str(ROOT / "outputs/stage1_class_selection/per_class_accuracy.csv"),
    "acc_C":      str(ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_C.csv"),
    "output_dir": str(ROOT / "outputs/analysis/dfa_mass_delta"),
}


def load_and_validate_accuracy(paths: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc_r = pd.read_csv(paths["acc_R"])
    acc_c = pd.read_csv(paths["acc_C"])
    if set(acc_r.columns) != set(acc_c.columns):
        raise ValueError(f"Accuracy files have different columns.\n  acc_R: {acc_r.columns.tolist()}\n  acc_C: {acc_c.columns.tolist()}")
    if "class_id" not in acc_r.columns:
        raise ValueError(f"No 'class_id' column found; got: {acc_r.columns.tolist()}")
    if "accuracy" not in acc_r.columns:
        raise ValueError(f"No 'accuracy' column found; got: {acc_r.columns.tolist()}")
    dfa_set = set(DFA_CLASSES)
    acc_r = acc_r[acc_r["class_id"].isin(dfa_set)].copy()
    acc_c = acc_c[acc_c["class_id"].isin(dfa_set)].copy()
    return acc_r, acc_c


def aggregate_per_class(clips: pd.DataFrame, acc_r: pd.DataFrame,
                        acc_c: pd.DataFrame) -> pd.DataFrame:
    grp = clips.groupby("class_id").agg(
        mean_delta=("delta", "mean"),
        n=("delta", "count"),
        sl_label=("sl_label", lambda x: x.mode()[0]),
    ).reset_index()
    acc = acc_r[["class_id", "accuracy"]].rename(columns={"accuracy": "acc_R"})
    acc = acc.merge(acc_c[["class_id", "accuracy"]].rename(columns={"accuracy": "acc_C"}),
                    on="class_id")
    acc["acc_drop"] = acc["acc_R"] - acc["acc_C"]
    return grp.merge(acc[["class_id", "acc_R", "acc_C", "acc_drop"]], on="class_id")


def compute_flip_rate(clips: pd.DataFrame) -> pd.Series:
    s_r = np.stack([np.asarray(v) for v in clips["signed_vec_R"]]).astype(np.float32)
    s_c = np.stack([np.asarray(v) for v in clips["signed_vec_C"]]).astype(np.float32)
    n_active   = (np.abs(s_r) > 1e-8).sum(axis=1).astype(float)
    flip_count = (np.sign(s_r) != np.sign(s_c)).sum(axis=1).astype(float)
    n_active[n_active == 0] = np.nan
    return pd.Series(flip_count / n_active, index=clips.index)


def plot_kde(clips: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x_range = np.linspace(clips["delta"].min(), clips["delta"].max(), 500)
    for label, colour in SL_COLOURS.items():
        vals = clips.loc[clips["sl_label"] == label, "delta"].values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        ax.fill_between(x_range, kde(x_range), alpha=0.4, color=colour,
                        label=f"{label.capitalize()} (n={len(vals)})")
        ax.plot(x_range, kde(x_range), color=colour, linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("delta  (total_abs_R − total_abs_C)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kde_delta_temporal_vs_static.png", dpi=150)
    plt.close(fig)
    print(f"  KDE → {out_dir / 'kde_delta_temporal_vs_static.png'}")


def plot_scatter(per_class: pd.DataFrame, out_dir: Path) -> None:
    n_min, n_max = per_class["n"].min(), per_class["n"].max()
    sizes = 30 + 170 * (per_class["n"] - n_min) / max(n_max - n_min, 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, colour in SL_COLOURS.items():
        mask = per_class["sl_label"] == label
        ax.scatter(per_class.loc[mask, "mean_delta"], per_class.loc[mask, "acc_drop"],
                   s=sizes[mask], c=colour, alpha=0.8, label=label.capitalize(), zorder=3)
    for _, row in per_class.iterrows():
        ax.annotate(str(int(row["class_id"])), (row["mean_delta"], row["acc_drop"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean delta  (R − C)")
    ax.set_ylabel("accuracy drop  (acc_R − acc_C)")
    ax.legend(title="point size = clip count")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_delta_vs_accdrop.png", dpi=150)
    plt.close(fig)
    print(f"  Scatter → {out_dir / 'scatter_delta_vs_accdrop.png'}")


def plot_kde_signed(clips: pd.DataFrame, out_dir: Path) -> None:
    clips = clips.copy()
    clips["signed_delta"] = clips["total_signed_R"] - clips["total_signed_C"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x_range = np.linspace(clips["signed_delta"].min(), clips["signed_delta"].max(), 500)
    for label, colour in SL_COLOURS.items():
        vals = clips.loc[clips["sl_label"] == label, "signed_delta"].values
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals)
        ax.fill_between(x_range, kde(x_range), alpha=0.4, color=colour,
                        label=f"{label.capitalize()} (n={len(vals)})")
        ax.plot(x_range, kde(x_range), color=colour, linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("signed delta  (total_signed_R − total_signed_C)")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "kde_signed_delta_temporal_vs_static.png", dpi=150)
    plt.close(fig)
    print(f"  KDE signed → {out_dir / 'kde_signed_delta_temporal_vs_static.png'}")


def plot_scatter_fliprate(clips: pd.DataFrame, per_class: pd.DataFrame, out_dir: Path) -> None:
    fr = clips.copy()
    fr["flip_rate"] = compute_flip_rate(clips)
    fr = fr.groupby("class_id")["flip_rate"].mean().reset_index()
    pc = per_class[["class_id", "sl_label", "acc_drop", "n"]].merge(fr, on="class_id")
    n_min, n_max = pc["n"].min(), pc["n"].max()
    sizes = 30 + 170 * (pc["n"] - n_min) / max(n_max - n_min, 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, colour in SL_COLOURS.items():
        mask = pc["sl_label"] == label
        ax.scatter(pc.loc[mask, "flip_rate"], pc.loc[mask, "acc_drop"],
                   s=sizes[mask], c=colour, alpha=0.8, label=label.capitalize(), zorder=3)
    for _, row in pc.iterrows():
        ax.annotate(str(int(row["class_id"])), (row["flip_rate"], row["acc_drop"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean flip rate  (sign changes / n_active features)")
    ax.set_ylabel("accuracy drop  (acc_R − acc_C)")
    ax.legend(title="point size = clip count")
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_fliprate_vs_accdrop.png", dpi=150)
    plt.close(fig)
    print(f"  Scatter flip rate → {out_dir / 'scatter_fliprate_vs_accdrop.png'}")


def main() -> None:
    out_dir = Path(PATHS["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_r, acc_c = load_and_validate_accuracy(PATHS)
    clips = pd.read_parquet(PATHS["parquet"])
    clips = clips[clips["class_id"].isin(set(DFA_CLASSES))]

    per_class = aggregate_per_class(clips, acc_r, acc_c)
    per_class.to_csv(out_dir / "per_class_summary.csv", index=False)
    print(f"  CSV → {out_dir / 'per_class_summary.csv'}")

    plot_kde(clips, out_dir)
    plot_scatter(per_class, out_dir)
    plot_kde_signed(clips, out_dir)
    plot_scatter_fliprate(clips, per_class, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
