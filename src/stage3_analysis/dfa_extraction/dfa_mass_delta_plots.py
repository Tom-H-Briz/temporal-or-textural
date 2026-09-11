"""
DFA mass delta visualisation — KDE and scatter plots.

Reads dfa_mass_delta parquet + per-class accuracy CSVs. No DFA engine calls.

CFG controls which parquet/condition to run — update per run.
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

CFG = {
    "parquet":    ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1.parquet",
    "acc_R":      ROOT / "outputs/stage1_class_selection/per_class_accuracy.csv",
    "acc_C":      ROOT / "outputs/stage1_class_selection_VM/per_class_accuracy_VM_C1.csv",
    "acc_A":      ROOT / "outputs/stage1_class_selection_VM/per_class_accuracy_VM_A.csv",
    "output_dir": ROOT / "outputs/analysis/dfa_mass_delta_vm_c1",
    "cond":       "C1",
    # --- TF equivalent ---
    # "parquet":    ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
    # "acc_C":      ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_C.csv",
    # "acc_A":      ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_A.csv",
    # "output_dir": ROOT / "outputs/analysis/dfa_mass_delta",
    # "cond":       "C",
}


def load_and_validate_accuracy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc_r = pd.read_csv(CFG["acc_R"])
    acc_c = pd.read_csv(CFG["acc_C"])
    acc_a = pd.read_csv(CFG["acc_A"])
    for name, df in [("acc_C", acc_c), ("acc_A", acc_a)]:
        if set(df.columns) != set(acc_r.columns):
            raise ValueError(f"{name} columns differ from acc_R: {df.columns.tolist()}")
    dfa_set = set(DFA_CLASSES)
    return (acc_r[acc_r["class_id"].isin(dfa_set)].copy(),
            acc_c[acc_c["class_id"].isin(dfa_set)].copy(),
            acc_a[acc_a["class_id"].isin(dfa_set)].copy())


def aggregate_per_class(clips: pd.DataFrame, acc_r: pd.DataFrame,
                        acc_c: pd.DataFrame, acc_a: pd.DataFrame) -> pd.DataFrame:
    grp = clips.groupby("class_id").agg(
        mean_delta=("delta", "mean"),
        mean_delta_A=("delta_A", "mean"),
        n=("delta", "count"),
        sl_label=("sl_label", lambda x: x.mode()[0]),
    ).reset_index()
    acc = acc_r[["class_id", "accuracy"]].rename(columns={"accuracy": "acc_R"})
    acc = acc.merge(acc_c[["class_id", "accuracy"]].rename(columns={"accuracy": "acc_C"}), on="class_id")
    acc = acc.merge(acc_a[["class_id", "accuracy"]].rename(columns={"accuracy": "acc_A"}), on="class_id")
    acc["acc_drop_C"] = acc["acc_R"] - acc["acc_C"]
    acc["acc_drop_A"] = acc["acc_R"] - acc["acc_A"]
    return grp.merge(acc[["class_id", "acc_R", "acc_C", "acc_A", "acc_drop_C", "acc_drop_A"]], on="class_id")


def compute_flip_rate(clips: pd.DataFrame, col_b: str) -> pd.Series:
    s_r = np.stack([np.asarray(v) for v in clips["signed_vec_R"]]).astype(np.float32)
    s_b = np.stack([np.asarray(v) for v in clips[col_b]]).astype(np.float32)
    n_active   = (np.abs(s_r) > 1e-8).sum(axis=1).astype(float)
    flip_count = (np.sign(s_r) != np.sign(s_b)).sum(axis=1).astype(float)
    n_active[n_active == 0] = np.nan
    return pd.Series(flip_count / n_active, index=clips.index)


def plot_kde(clips: pd.DataFrame, out_dir: Path) -> None:
    cond     = CFG["cond"]
    all_vals = np.concatenate([clips["delta"].values, clips["delta_A"].values])
    x_range  = np.linspace(all_vals.min(), all_vals.max(), 500)
    fig, ax  = plt.subplots(figsize=(9, 5))
    for label, colour in SL_COLOURS.items():
        mask = clips["sl_label"] == label
        for col, ls, c in [("delta", "-", cond), ("delta_A", "--", "A")]:
            vals = clips.loc[mask, col].values
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals)
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=colour)
            ax.plot(x_range, kde(x_range), color=colour, linewidth=1.5,
                    linestyle=ls, label=f"{label.capitalize()} R−{c} (n={len(vals)})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"delta  (total_abs_R − total_abs_X)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "kde_delta_temporal_vs_static.png", dpi=150)
    plt.close(fig)
    print(f"  KDE → {out_dir / 'kde_delta_temporal_vs_static.png'}")


def plot_scatter(per_class: pd.DataFrame, out_dir: Path) -> None:
    cond   = CFG["cond"]
    n_min, n_max = per_class["n"].min(), per_class["n"].max()
    sizes  = 30 + 170 * (per_class["n"] - n_min) / max(n_max - n_min, 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, colour in SL_COLOURS.items():
        mask = per_class["sl_label"] == label
        ax.scatter(per_class.loc[mask, "mean_delta"], per_class.loc[mask, "acc_drop_C"],
                   s=sizes[mask], c=colour, alpha=0.8, marker="o",
                   label=f"{label.capitalize()} R−{cond}", zorder=3)
        ax.scatter(per_class.loc[mask, "mean_delta_A"], per_class.loc[mask, "acc_drop_A"],
                   s=sizes[mask], c=colour, alpha=0.8, marker="^",
                   label=f"{label.capitalize()} R−A", zorder=3)
    for _, row in per_class.iterrows():
        ax.annotate(str(int(row["class_id"])), (row["mean_delta"], row["acc_drop_C"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean delta  (R − X)")
    ax.set_ylabel("accuracy drop  (acc_R − acc_X)")
    ax.legend(title=f"○={cond}  △=A", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_delta_vs_accdrop.png", dpi=150)
    plt.close(fig)
    print(f"  Scatter → {out_dir / 'scatter_delta_vs_accdrop.png'}")


def plot_kde_signed(clips: pd.DataFrame, out_dir: Path) -> None:
    cond  = CFG["cond"]
    clips = clips.copy()
    clips["signed_delta_C"] = clips["total_signed_R"] - clips[f"total_signed_{cond}"]
    clips["signed_delta_A"] = clips["total_signed_R"] - clips["total_signed_A"]
    all_vals = np.concatenate([clips["signed_delta_C"].values, clips["signed_delta_A"].values])
    x_range  = np.linspace(all_vals.min(), all_vals.max(), 500)
    fig, ax  = plt.subplots(figsize=(9, 5))
    for label, colour in SL_COLOURS.items():
        mask = clips["sl_label"] == label
        for col, ls, c in [("signed_delta_C", "-", cond), ("signed_delta_A", "--", "A")]:
            vals = clips.loc[mask, col].values
            if len(vals) < 2:
                continue
            kde = gaussian_kde(vals)
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=colour)
            ax.plot(x_range, kde(x_range), color=colour, linewidth=1.5,
                    linestyle=ls, label=f"{label.capitalize()} R−{c} (n={len(vals)})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("signed delta  (total_signed_R − total_signed_X)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "kde_signed_delta_temporal_vs_static.png", dpi=150)
    plt.close(fig)
    print(f"  KDE signed → {out_dir / 'kde_signed_delta_temporal_vs_static.png'}")


def plot_scatter_fliprate(clips: pd.DataFrame, per_class: pd.DataFrame, out_dir: Path) -> None:
    cond = CFG["cond"]
    fr   = clips.copy()
    fr["flip_rate_C"] = compute_flip_rate(clips, col_b=f"signed_vec_{cond}")
    fr["flip_rate_A"] = compute_flip_rate(clips, col_b="signed_vec_A")
    fr = fr.groupby("class_id")[["flip_rate_C", "flip_rate_A"]].mean().reset_index()
    pc = per_class[["class_id", "sl_label", "acc_drop_C", "acc_drop_A", "n"]].merge(fr, on="class_id")
    n_min, n_max = pc["n"].min(), pc["n"].max()
    sizes = 30 + 170 * (pc["n"] - n_min) / max(n_max - n_min, 1)
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, colour in SL_COLOURS.items():
        mask = pc["sl_label"] == label
        ax.scatter(pc.loc[mask, "flip_rate_C"], pc.loc[mask, "acc_drop_C"],
                   s=sizes[mask], c=colour, alpha=0.8, marker="o",
                   label=f"{label.capitalize()} R−{cond}", zorder=3)
        ax.scatter(pc.loc[mask, "flip_rate_A"], pc.loc[mask, "acc_drop_A"],
                   s=sizes[mask], c=colour, alpha=0.8, marker="^",
                   label=f"{label.capitalize()} R−A", zorder=3)
    for _, row in pc.iterrows():
        ax.annotate(str(int(row["class_id"])), (row["flip_rate_C"], row["acc_drop_C"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("mean flip rate  (sign changes / n_active features)")
    ax.set_ylabel("accuracy drop  (acc_R − acc_X)")
    ax.legend(title=f"○={cond}  △=A", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_fliprate_vs_accdrop.png", dpi=150)
    plt.close(fig)
    print(f"  Scatter flip rate → {out_dir / 'scatter_fliprate_vs_accdrop.png'}")


def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    acc_r, acc_c, acc_a = load_and_validate_accuracy()
    clips = pd.read_parquet(CFG["parquet"])
    clips = clips[clips["class_id"].isin(set(DFA_CLASSES))]
    clips["delta_A"] = clips["total_abs_R"] - clips["total_abs_A"]

    per_class = aggregate_per_class(clips, acc_r, acc_c, acc_a)
    per_class.to_csv(out_dir / "per_class_summary.csv", index=False)
    print(f"  CSV → {out_dir / 'per_class_summary.csv'}")

    plot_kde(clips, out_dir)
    plot_scatter(per_class, out_dir)
    plot_kde_signed(clips, out_dir)
    plot_scatter_fliprate(clips, per_class, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
