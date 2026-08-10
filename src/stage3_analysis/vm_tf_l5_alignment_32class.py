"""
32-class table: Real TF, Real VM, VM L5-ablated — and the alignment between them.

Population is TF's original 32-class DFA pilot (dfa_mass_delta.py's
CFG["dfa_classes"], reused directly, not retyped) — the 35-class ablation-
impact population minus the 3 BAD_CLASSES (38, 97, 160).

vm_accuracy_l5_ablated rescales the L5-ablation population's survival rate
(all7 scaffold, R condition) back onto vm_accuracy's full denominator, same
method as vm_tf_accuracy_vs_l5_ablation.py — not the clip-paired-bootstrap
version, this is the plain point-estimate table.

alignment = |vm_accuracy_l5_ablated - vm_accuracy| / (vm_accuracy - tf_accuracy)
— does VM's own L5-ablation-caused accuracy drop match the size of its lead
over TF?

Output:
    outputs/analysis/scaffold_ablation/vm_tf_l5_alignment_32class.csv

Usage:
    uv run python src/stage3_analysis/vm_tf_l5_alignment_32class.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.dfa_mass_delta import CFG as TF_DFA_CFG

OUT_DIR = ROOT / "outputs" / "analysis" / "scaffold_ablation"
L5_PARQUET = OUT_DIR / "ablation_results_long_l5_job7ep_k64.parquet"
L5_TARGET = "all7"
VM_ACC_CSV = ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv"
TF_ACC_CSV = ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv"
CLASSES = TF_DFA_CFG["dfa_classes"]


def load_real_scores() -> pd.DataFrame:
    vm = pd.read_csv(VM_ACC_CSV)[["class_id", "template", "accuracy"]].rename(columns={"accuracy": "vm_real"})
    tf = pd.read_csv(TF_ACC_CSV)[["class_id", "accuracy"]].rename(columns={"accuracy": "tf_real"})
    out = vm.merge(tf, on="class_id")
    return out[out["class_id"].isin(CLASSES)]


def load_l5_survival_rate() -> pd.DataFrame:
    df = pd.read_parquet(L5_PARQUET, columns=["class_id", "perturbation_condition",
                                              "ablation_target", "correct_ablated"])
    df = df[(df["perturbation_condition"] == "R") & (df["ablation_target"] == L5_TARGET)]
    return df.groupby("class_id")["correct_ablated"].mean().rename("l5_survival_rate").reset_index()


def build_table() -> pd.DataFrame:
    df = load_real_scores().merge(load_l5_survival_rate(), on="class_id")
    df["vm_l5_ablated"] = df["vm_real"] * df["l5_survival_rate"]
    df["accuracy_delta_unablated"] = df["vm_real"] - df["tf_real"]
    df["accuracy_delta_l5"] = df["vm_l5_ablated"] - df["vm_real"]
    df["alignment"] = df["accuracy_delta_l5"].abs() / df["accuracy_delta_unablated"]
    cols = ["class_id", "template", "tf_real", "vm_real", "vm_l5_ablated",
            "accuracy_delta_unablated", "accuracy_delta_l5", "alignment"]
    return df[cols].sort_values("class_id").reset_index(drop=True)


def main() -> None:
    out = build_table()
    assert len(out) == len(CLASSES), f"expected {len(CLASSES)} classes, got {len(out)}"
    out_path = OUT_DIR / "vm_tf_l5_alignment_32class.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} classes -> {out_path}")
    print(f"median alignment: {out['alignment'].median():.3f}  mean: {out['alignment'].mean():.3f}")


if __name__ == "__main__":
    main()
