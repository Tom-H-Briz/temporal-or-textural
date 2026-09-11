"""
Exploratory analysis of TF cumulative_mass_diagnostic parquets.

Reads class labels from accuracy_SL_subset.csv (Laura SL temporal/static).
SAE_LAYER env var selects which layer's parquet to analyse.

Outputs (to outputs/analysis/explore/):
  n90_vs_logit_margin_tf_l{N}.png
  per_class_n90_tf_l{N}.png
  per_class_summary_tf_l{N}.csv
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

SL_COLOURS = {
    "temporal":   "steelblue",
    "static":     "darkorange",
    "both":       "mediumpurple",
    "unlabelled": "lightgrey",
}

_layer = int(os.environ.get("SAE_LAYER", 7))
_k, _x = 64, 8

CFG = {
    "layer":               _layer,
    "parquet_path":        str(ROOT / f"outputs/analysis/cumulative_mass_diagnostic_tf_l{_layer}_k{_k}_x{_x}.parquet"),
    "sl_csv_path":         str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_suffix":       f"_tf_l{_layer}",
    "output_dir":          str(ROOT / "outputs/analysis/explore/"),
    "min_clips_per_class": 5,
    "dev_class_ids":       [],
}


def build_sl_label_map(cfg: dict) -> dict[int, str]:
    df = pd.read_csv(cfg["sl_csv_path"])
    return {int(row["class_id"]): row["category"] for _, row in df.iterrows()}


def make_scatter(df: pd.DataFrame, sl_map: dict, cfg: dict, out_dir: Path, suffix: str) -> None:
    dev_ids = set(cfg["dev_class_ids"])
    df = df.copy()
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    df["is_dev"]   = df["class_id"].isin(dev_ids)
    fig, ax = plt.subplots(figsize=(9, 6))
    for is_dev, ms, zo in [(False, 10, 1), (True, 60, 2)]:
        sub = df[df["is_dev"] == is_dev]
        for label, colour in SL_COLOURS.items():
            grp = sub[sub["sl_label"] == label]
            if grp.empty:
                continue
            ax.scatter(grp["logit_margin"], grp["n90"], c=colour, s=ms,
                       alpha=0.5 if not is_dev else 0.9,
                       linewidths=0.5 if is_dev else 0,
                       edgecolors="black" if is_dev else "none",
                       label=label if not is_dev else None, zorder=zo)
    ax.set_xlabel("Logit margin")
    ax.set_ylabel("N90")
    ax.set_title(f"N90 vs logit margin  (TF layer {cfg['layer']})")
    ax.legend(title="SL label", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / f"n90_vs_logit_margin{suffix}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_dir / f'n90_vs_logit_margin{suffix}.png'}")


def make_bar_chart(df: pd.DataFrame, sl_map: dict, cfg: dict, out_dir: Path, suffix: str) -> None:
    stats = (df.groupby(["class_id", "class_name"])["n90"]
             .agg(n_clips="count", median=lambda x: x.median(),
                  p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
             .reset_index())
    stats = stats[stats["n_clips"] >= cfg["min_clips_per_class"]]
    stats["sl_label"] = stats["class_id"].map(sl_map).fillna("unlabelled")
    stats = stats.sort_values("median").reset_index(drop=True)
    n = len(stats)
    fig, ax = plt.subplots(figsize=(10, max(8, n * 0.22)))
    colours = [SL_COLOURS[lbl] for lbl in stats["sl_label"]]
    xerr_lo = (stats["median"] - stats["p25"]).values
    xerr_hi = (stats["p75"] - stats["median"]).values
    ax.barh(np.arange(n), stats["median"], xerr=[xerr_lo, xerr_hi],
            color=colours, alpha=0.8, error_kw={"linewidth": 0.6, "capsize": 2})
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(stats["class_name"], fontsize=5)
    ax.set_xlabel("Median N90  (error bars = IQR)")
    ax.set_title(f"Per-class median N90  (TF layer {cfg['layer']}, n≥{cfg['min_clips_per_class']})")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in SL_COLOURS.items()],
              title="SL label", loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"per_class_n90{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_dir / f'per_class_n90{suffix}.png'}")


def make_csv(df: pd.DataFrame, sl_map: dict, out_dir: Path, suffix: str) -> None:
    med = "median"
    agg = (df.groupby(["class_id", "class_name"])
           .agg(
               n_clips               = ("n90",            "count"),
               n50_median            = ("n50",            med),
               n80_median            = ("n80",            med),
               n90_median            = ("n90",            med),
               n90_iqr               = ("n90",            lambda x: x.quantile(0.75) - x.quantile(0.25)),
               n95_median            = ("n95",            med),
               n_active_median       = ("n_active",       med),
               max_dfa_median        = ("max_dfa",        med),
               mean_dfa_median       = ("mean_dfa",       med),
               max_mean_ratio_median = ("max_mean_ratio", med),
               top1_frac_median      = ("top1_frac",      med),
               gini_median           = ("gini",           med),
               suppressor_frac_median= ("suppressor_frac",med),
               logit_margin_median   = ("logit_margin",   med),
               softmax_entropy_median= ("softmax_entropy",med),
           )
           .reset_index())
    agg["sl_label"] = agg["class_id"].map(sl_map).fillna("unlabelled")
    agg = agg.sort_values("class_id").reset_index(drop=True)
    agg.to_csv(out_dir / f"per_class_summary{suffix}.csv", index=False)
    print(f"  Saved → {out_dir / f'per_class_summary{suffix}.csv'}  ({len(agg)} rows)")


def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CFG["parquet_path"])
    print(f"Loaded {len(df):,} clips, {df['class_id'].nunique()} classes  (layer {CFG['layer']})")
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
