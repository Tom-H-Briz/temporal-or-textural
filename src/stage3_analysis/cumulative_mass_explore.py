"""
Exploratory analysis of cumulative_mass_diagnostic.parquet.

Outputs:
  outputs/analysis/explore/n90_vs_logit_margin.png
  outputs/analysis/explore/per_class_n90.png
  outputs/analysis/explore/per_class_summary.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CFG = {
    "parquet_path": str(ROOT / "outputs/analysis/cumulative_mass_diagnostic_16x.parquet"),
    "output_suffix": "_16x",        # e.g. "_16x" to compare SAE sizes
    "output_dir": str(ROOT / "outputs/analysis/explore/"),
    "min_clips_per_class": 5,
    "sevilla_lara_temporal": [1, 6, 10, 12, 26, 34, 39, 45, 47, 56, 60, 63, 67, 70, 77, 81, 95, 150],
    "sevilla_lara_static":   [9, 15, 42, 50, 58, 61, 63, 71, 78, 90, 93, 106, 109, 114, 121, 130, 138, 165],
    "dev_class_ids": [93, 94, 97, 149, 150],
}

SL_COLOURS = {
    "temporal":   "steelblue",
    "static":     "darkorange",
    "both":       "mediumpurple",
    "unlabelled": "lightgrey",
}


def build_sl_label_map(cfg: dict) -> dict[int, str]:
    temporal = set(cfg["sevilla_lara_temporal"])
    static   = set(cfg["sevilla_lara_static"])
    label_map: dict[int, str] = {}
    for cid in temporal | static:
        if cid in temporal and cid in static:
            label_map[cid] = "both"
        elif cid in temporal:
            label_map[cid] = "temporal"
        else:
            label_map[cid] = "static"
    return label_map


def make_scatter(df: pd.DataFrame, sl_map: dict[int, str], cfg: dict, out_dir: Path, suffix: str) -> None:
    dev_ids = set(cfg["dev_class_ids"])
    df = df.copy()
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    df["is_dev"]   = df["class_id"].isin(dev_ids)

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot non-dev clips first (background), then dev clips on top
    for is_dev, marker_size, zorder in [(False, 10, 1), (True, 60, 2)]:
        subset = df[df["is_dev"] == is_dev]
        for label, colour in SL_COLOURS.items():
            group = subset[subset["sl_label"] == label]
            if group.empty:
                continue
            ax.scatter(
                group["logit_margin"], group["n90"],
                c=colour, s=marker_size, alpha=0.5 if not is_dev else 0.9,
                linewidths=0.5 if is_dev else 0,
                edgecolors="black" if is_dev else "none",
                label=label if not is_dev else None,
                zorder=zorder,
            )

    ax.set_xlabel("Logit margin")
    ax.set_ylabel("N90")
    ax.set_title("N90 vs logit margin")
    ax.legend(title="Sevilla-Lara", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"n90_vs_logit_margin{suffix}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / f'n90_vs_logit_margin{suffix}.png'}")


def make_bar_chart(df: pd.DataFrame, sl_map: dict[int, str], cfg: dict, out_dir: Path, suffix: str) -> None:
    dev_ids = set(cfg["dev_class_ids"])

    stats = (
        df.groupby(["class_id", "class_name"])["n90"]
        .agg(
            n_clips="count",
            median=lambda x: x.median(),
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    stats = stats[stats["n_clips"] >= cfg["min_clips_per_class"]]
    stats["sl_label"] = stats["class_id"].map(sl_map).fillna("unlabelled")
    stats = stats.sort_values("median").reset_index(drop=True)

    n = len(stats)
    fig_height = max(8, n * 0.22)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(n)
    colours = [SL_COLOURS[lbl] for lbl in stats["sl_label"]]
    xerr_lo = (stats["median"] - stats["p25"]).values
    xerr_hi = (stats["p75"]    - stats["median"]).values

    ax.barh(
        y_pos, stats["median"], xerr=[xerr_lo, xerr_hi],
        color=colours, alpha=0.8, error_kw={"linewidth": 0.6, "capsize": 2},
    )

    # Annotate dev classes with asterisk
    for i, row in stats.iterrows():
        idx = stats.index.get_loc(i)
        if row["class_id"] in dev_ids:
            ax.text(
                stats["median"].max() * 1.01, idx, "*",
                va="center", fontsize=8, fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stats["class_name"], fontsize=5)
    ax.set_xlabel("Median N90  (error bars = IQR)")
    ax.set_title(f"Per-class median N90  (n≥{cfg['min_clips_per_class']})")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=l) for l, c in SL_COLOURS.items()]
    ax.legend(handles=legend_handles, title="Sevilla-Lara", loc="lower right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / f"per_class_n90{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_dir / f'per_class_n90{suffix}.png'}")


def make_csv(df: pd.DataFrame, sl_map: dict[int, str], out_dir: Path, suffix: str) -> None:
    agg = (
        df.groupby(["class_id", "class_name"])
        .agg(
            n_clips=("n90", "count"),
            n90_median=("n90", "median"),
            n90_iqr=("n90", lambda x: x.quantile(0.75) - x.quantile(0.25)),
            n90_min=("n90", "min"),
            n90_max=("n90", "max"),
            gini_median=("gini", "median"),
            max_mean_ratio_median=("max_mean_ratio", "median"),
            logit_median=("logit", "median"),
            suppressor_frac_median=("suppressor_frac", "median"),
        )
        .reset_index()
    )
    agg["sevilla_lara_label"] = agg["class_id"].map(sl_map).fillna("unlabelled")
    agg = agg.sort_values("class_id").reset_index(drop=True)
    agg.to_csv(out_dir / f"per_class_summary{suffix}.csv", index=False)
    print(f"  Saved → {out_dir / f'per_class_summary{suffix}.csv'}  ({len(agg)} rows)")


def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CFG["parquet_path"])
    print(f"Loaded {len(df):,} clips, {df['class_id'].nunique()} classes")

    sl_map = build_sl_label_map(CFG)

    suffix = CFG["output_suffix"]

    print("Scatter plot...")
    make_scatter(df, sl_map, CFG, out_dir, suffix)

    print("Bar chart...")
    make_bar_chart(df, sl_map, CFG, out_dir, suffix)

    print("Summary CSV...")
    make_csv(df, sl_map, out_dir, suffix)


if __name__ == "__main__":
    main()
