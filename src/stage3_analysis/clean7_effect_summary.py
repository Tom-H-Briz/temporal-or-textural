"""
Clean7 ablation effect summary — per class with human-readable labels.

Sources:
    ablation_results_long_clean7_020726.parquet  → median logit delta (clean7, R)
    ablation_margin_sl_all_clean7_020726.csv     → flip rate
    data/ssv2/labels/labels.json                → class names

Output: outputs/analysis/scaffold_ablation/clean7_effect_summary.csv

Usage:
    uv run python src/stage3_analysis/clean7_effect_summary.py
"""

from pathlib import Path
import sys, json

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
from ToT_utils import load_metadata, _strip_brackets

BAD_CLASSES = {38, 83, 97, 160}

CFG = {
    "ablation_pq":  ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_clean7_020726.parquet",
    "margin_csv":   ROOT / "outputs/analysis/scaffold_ablation/ablation_margin_sl_all_clean7_020726.csv",
    "labels_path":  ROOT / "data/ssv2/labels/labels.json",
    "val_path":     ROOT / "data/ssv2/labels/validation.json",
    "acc_csv":      ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "out":          ROOT / "outputs/analysis/scaffold_ablation/clean7_effect_summary.csv",
}


def load_class_names(cfg: dict) -> dict[int, str]:
    label_map, _, _ = load_metadata(str(cfg["labels_path"]), str(cfg["val_path"]))
    return {v: k for k, v in label_map.items()}


def main() -> None:
    acc    = pd.read_csv(CFG["acc_csv"])
    valid  = set(acc.loc[acc["accuracy"] >= 0.40, "class_id"])
    names  = load_class_names(CFG)

    # Margin pct change and flip rate — from margin CSV (clean7, R condition)
    margin = pd.read_csv(CFG["margin_csv"])
    margin = margin[margin["true_class"].isin(valid)]
    margin_agg = margin.groupby(["true_class","sl_label"]).agg(
        n_clips=("clip_id","count"),
        median_margin_pct_change=("margin_r1r2_pct_change","median"),
        flip_rate=("flip","mean"),
    ).reset_index().rename(columns={"true_class":"class_id"})

    # Join and add names
    out = margin_agg.copy()
    out["class_name"] = out["class_id"].map(names)
    out = out[["class_id","class_name","sl_label","n_clips",
               "median_margin_pct_change","flip_rate"]]
    out = out.sort_values("median_margin_pct_change").reset_index(drop=True)

    out.to_csv(CFG["out"], index=False)
    print(f"  {len(out)} classes → {CFG['out']}")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
