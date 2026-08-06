"""
Position-locked ablation impact vs. VM/TF accuracy gap on SSv2.

Do the classes most disrupted by ablating VM's position-locked scaffold
features (L5 all7, L7 all4 — ablation_l5_vs_l7_class_impact.csv, already
cached) overlap with the classes where VM and TF diverge most in R-accuracy?
If they do, that's evidence VM's positional reliance is specifically what's
driving the VM-TF performance gap on those classes, not a coincidence.

Outputs (outputs/analysis/scaffold_ablation/):
    position_lock_vs_backbone_gap.csv

Usage:
    uv run python src/stage3_analysis/position_lock_vs_backbone_gap.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "analysis" / "scaffold_ablation"
IMPACT_CSV = OUT_DIR / "ablation_l5_vs_l7_class_impact.csv"
VM_ACC_CSV = ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv"
TF_ACC_CSV = ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv"


def load_accuracy_gap() -> pd.DataFrame:
    """Full 174-class VM/TF R-accuracy, joined and signed VM-minus-TF —
    positive means VM beats TF on that class, negative means TF beats VM."""
    vm = pd.read_csv(VM_ACC_CSV)[["class_id", "accuracy"]].rename(columns={"accuracy": "vm_accuracy"})
    tf = pd.read_csv(TF_ACC_CSV)[["class_id", "accuracy"]].rename(columns={"accuracy": "tf_accuracy"})
    merged = vm.merge(tf, on="class_id")
    merged["vm_minus_tf"] = merged["vm_accuracy"] - merged["tf_accuracy"]
    merged["abs_gap"] = merged["vm_minus_tf"].abs()
    return merged


IMPACT_COLS = ["mean_delta_l5", "mean_delta_l7", "flip_rate_l5", "flip_rate_l7"]


def report_correlations(merged: pd.DataFrame) -> None:
    print(f"\nSpearman correlations (n={len(merged)} classes):")
    for col in IMPACT_COLS:
        r_signed = stats.spearmanr(merged[col], merged["vm_minus_tf"])
        r_abs = stats.spearmanr(merged[col], merged["abs_gap"])
        print(f"  {col:16s} vs vm_minus_tf: rho={r_signed.statistic:+.3f} (p={r_signed.pvalue:.4f})  |  "
              f"vs abs_gap: rho={r_abs.statistic:+.3f} (p={r_abs.pvalue:.4f})")


def report_top_overlap(merged: pd.DataFrame, impact_col: str, n: int = 10) -> None:
    top_gap = set(merged.nlargest(n, "abs_gap")["class_id"])
    top_impact = set(merged.nlargest(n, impact_col)["class_id"])
    overlap = top_gap & top_impact
    print(f"\nTop-{n} by abs_gap vs top-{n} by {impact_col}: {len(overlap)} shared classes -> {sorted(overlap)}")


def main() -> None:
    impact = pd.read_csv(IMPACT_CSV)
    gap = load_accuracy_gap()
    merged = impact.merge(gap, on="class_id")
    print(f"{len(merged)} classes with both position-locked ablation impact and VM/TF accuracy data")

    report_correlations(merged)
    for col in ("mean_delta_l5", "mean_delta_l7"):
        report_top_overlap(merged, col)

    merged = merged.sort_values("abs_gap", ascending=False)
    merged.to_csv(OUT_DIR / "position_lock_vs_backbone_gap.csv", index=False)


if __name__ == "__main__":
    main()
