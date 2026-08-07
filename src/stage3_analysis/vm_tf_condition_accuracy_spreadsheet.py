"""
Per-class accuracy spreadsheet — VM and TF, R/perturbation conditions.

VM: R, C1 (paired-shuffle), A (midpoint-frame-repeated).
TF: R, C (full-permutation shuffle), A (midpoint-frame-repeated).
Different shuffle mechanisms per backbone (C1 vs C) by design — see
run_ablation_tf.py's docstring for why they aren't interchangeable.

Reuses the already-cached per_class_accuracy_*.csv files from
notebooks/perturb_accuracy_vm_ssv2.py / perturb_accuracy_tf.py — no new
inference. Restricted to the 35-class SL-pilot population already
established this session (vm_tf_accuracy_vs_l5_ablation.csv).

Output:
    outputs/analysis/scaffold_ablation/vm_tf_condition_accuracy.xlsx

Usage:
    uv run python src/stage3_analysis/vm_tf_condition_accuracy_spreadsheet.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "analysis" / "scaffold_ablation"
CLASS_SOURCE = OUT_DIR / "vm_tf_accuracy_vs_l5_ablation.csv"
VM_DIR = ROOT / "outputs/stage1_class_selection_VM_ssv2"
TF_DIR = ROOT / "outputs/stage1_class_selection_TF"

SOURCES = {
    "vm_r_accuracy":  VM_DIR / "per_class_accuracy_VM_ssv2_R.csv",
    "vm_c1_accuracy": VM_DIR / "per_class_accuracy_VM_ssv2_C1.csv",
    "vm_a_accuracy":  VM_DIR / "per_class_accuracy_VM_ssv2_A.csv",
    "tf_r_accuracy":  TF_DIR / "per_class_accuracy_TF.csv",
    "tf_c_accuracy":  TF_DIR / "per_class_accuracy_TF_C.csv",
    "tf_a_accuracy":  TF_DIR / "per_class_accuracy_TF_A.csv",
}


def load_class_list() -> pd.DataFrame:
    """The 35-class SL-pilot population already used throughout this session's
    VM/TF comparisons — same class set, not a fresh re-derivation."""
    return pd.read_csv(CLASS_SOURCE, usecols=["class_id", "template"])


def build_table() -> pd.DataFrame:
    out = load_class_list()
    for col_name, path in SOURCES.items():
        src = pd.read_csv(path)[["class_id", "accuracy"]].rename(columns={"accuracy": col_name})
        out = out.merge(src, on="class_id", how="left")
    return out.sort_values("class_id").reset_index(drop=True)


def main() -> None:
    out = build_table()
    missing = out[out.isna().any(axis=1)]
    if len(missing):
        print(f"  WARNING: {len(missing)} classes missing at least one condition: "
              f"{missing['class_id'].tolist()}")
    out_path = OUT_DIR / "vm_tf_condition_accuracy.xlsx"
    out.to_excel(out_path, index=False, sheet_name="vm_tf_conditions")
    print(f"  {len(out)} classes -> {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
