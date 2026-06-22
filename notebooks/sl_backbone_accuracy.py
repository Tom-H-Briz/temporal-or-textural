"""
Raw per-backbone accuracy over the SL temporal/static class subset.
No SAE, no splice — straight model accuracy from per_class_accuracy CSVs.

Usage: uv run python notebooks/sl_backbone_accuracy.py
"""

from pathlib import Path
import pandas as pd

ROOT   = Path(__file__).parent.parent
SL_CSV = ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"
VMAE   = ROOT / "outputs/stage1_class_selection/per_class_accuracy.csv"
TF     = ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv"

sl     = pd.read_csv(SL_CSV)[["class_id", "category"]]
vmae   = pd.read_csv(VMAE).merge(sl, on="class_id")
tf     = pd.read_csv(TF).merge(sl,   on="class_id")


def report(df: pd.DataFrame, name: str) -> None:
    def acc(subset): return subset["correct"].sum() / subset["total"].sum()
    print(f"\n{name}")
    print(f"  SL overall:  {acc(df):.4f}  ({len(df)} classes)")
    print(f"  SL temporal: {acc(df[df['category']=='temporal']):.4f}")
    print(f"  SL static:   {acc(df[df['category']=='static']):.4f}")


report(vmae, "VideoMAE")
report(tf,   "TimeSformer")
