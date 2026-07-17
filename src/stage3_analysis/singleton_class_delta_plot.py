"""
Singleton per-class median logit delta — features 6135 and 2197, R condition.

Two-stage: persist per-clip table, then aggregate (median only — no mean).
Produces two bar charts (one per feature) for presentation.

Outputs (outputs/analysis/scaffold_ablation/):
    singleton_class_delta_percli_080726.parquet
    singleton_class_delta_6135_080726.csv
    singleton_class_delta_2197_080726.csv
    singleton_class_delta_6135_080726.png
    singleton_class_delta_2197_080726.png

Usage:
    uv run python src/stage3_analysis/singleton_class_delta_plot.py
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
from ToT_utils import load_metadata, _strip_brackets

FEATURES   = ["single_6135", "single_2197"]
HIGHLIGHT  = {6, 171}
BAD_CLASSES = {38, 83, 97, 160}

CFG = {
    "src_glob":    "outputs/analysis/scaffold_ablation/**/ablation_results_long_cover_uncover_060726.parquet",
    "acc_csv":     ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "val_path":    ROOT / "data/ssv2/labels/validation.json",
    "out_dir":     ROOT / "outputs/analysis/scaffold_ablation",
}


def load_class_names(cfg: dict) -> dict[int, str]:
    label_map, _, _ = load_metadata(str(cfg["labels_path"]), str(cfg["val_path"]))
    return {v: k for k, v in label_map.items()}


def load_per_clip(cfg: dict) -> pd.DataFrame:
    matches = list(ROOT.glob(cfg["src_glob"]))
    assert len(matches) == 1, f"Expected 1 parquet, found: {matches}"
    df = pd.read_parquet(matches[0],
                         columns=["clip_id", "class_id", "sl_label",
                                  "ablation_target", "perturbation_condition", "delta"])
    acc  = pd.read_csv(cfg["acc_csv"])
    valid = set(acc.loc[acc["accuracy"] >= 0.40, "class_id"])
    return df[
        (df["perturbation_condition"] == "R") &
        (df["ablation_target"].isin(FEATURES)) &
        (df["class_id"].isin(valid))
    ][["clip_id", "class_id", "sl_label", "ablation_target", "delta"]].copy()


def make_chart(agg: pd.DataFrame, feature: str, class_names: dict, out_path: Path) -> None:
    agg = agg.sort_values("median_delta", ascending=False).reset_index(drop=True)
    labels = [class_names.get(int(c), str(c)) for c in agg["class_id"]]
    colours = ["#C0392B" if int(c) in HIGHLIGHT else "#2980B9" for c in agg["class_id"]]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(agg)), agg["median_delta"], color=colours, width=0.7)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Median logit delta (baseline − ablated)", fontsize=9)
    ax.set_title(f"Singleton ablation — f{feature.split('_')[1]} — per-class median logit damage (R)", fontsize=10)

    legend = [
        mpatches.Patch(color="#C0392B", label="Class 6 / 171"),
        mpatches.Patch(color="#2980B9", label="Other SL classes"),
    ]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main() -> None:
    class_names = load_class_names(CFG)
    out_dir: Path = CFG["out_dir"]

    per_clip = load_per_clip(CFG)
    per_clip.to_parquet(out_dir / "singleton_class_delta_perclip_080726.parquet", index=False)
    print(f"Per-clip: {len(per_clip):,} rows → singleton_class_delta_perclip_080726.parquet")

    for feat in FEATURES:
        sub = per_clip[per_clip["ablation_target"] == feat]
        agg = sub.groupby(["class_id", "sl_label"]).agg(
            n_clips=("delta", "count"),
            median_delta=("delta", "median"),
        ).reset_index().sort_values("median_delta", ascending=False)

        tag = feat.split("_")[1]
        agg.to_csv(out_dir / f"singleton_class_delta_{tag}_080726.csv", index=False)
        print(f"Aggregated ({feat}): {len(agg)} classes → singleton_class_delta_{tag}_080726.csv")
        print(agg[["class_id", "sl_label", "n_clips", "median_delta"]].head(10).to_string(index=False))
        print()

        make_chart(agg, feat, class_names, out_dir / f"singleton_class_delta_{tag}_080726.png")


if __name__ == "__main__":
    main()
