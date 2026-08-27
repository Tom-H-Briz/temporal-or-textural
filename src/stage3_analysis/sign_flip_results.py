"""
Sign-flip / scaffold-ablation flip rate and accuracy drop — VM (L5, L7,
L5+L7 cross-layer) and TF (TOP12 sign-flip set), consolidated into one table.

Definitions (matching ablation_summary_tf_signflip.py's existing convention):
  flip_rate     = 1 - ablated_accuracy, on the R-eligibility-gated clip pool,
                   regardless of which condition is ablated under. Equals
                   accuracy_drop under R (baseline is 100% by construction of
                   the R-correct gate) but NOT under a shuffle condition
                   (C1/C), where the unablated baseline is <100% — flip_rate
                   there mixes shuffle damage + ablation damage, both
                   relative to the *original* R-correct baseline.
  accuracy_drop = unablated_accuracy(condition) - ablated_accuracy(condition),
                   both measured under the SAME condition. Isolates
                   ablation's marginal cost specifically, on top of whatever
                   the perturbation alone already costs.

Baseline sourcing:
  - VM: computed directly from this study's own stored baseline_all_logits
    (argmax vs class_id) — self-contained, no join needed.
  - TF: run_ablation_tf.py discards the unablated-C "correct" flag and never
    stores a full logit vector, so this joins outputs/analysis/dfa_mass_delta/
    dfa_mass_delta.parquet's correct_C column instead — confirmed exact
    clip_id match (3,656/3,656) with the ablation study's own R-eligible pool.

Known caveat, VM L7: the "all4" joint target is 3 gate-passed scaffold
members (5165/6021/6032) PLUS one deliberately-included near-miss (3347) —
see ablation_targets.py's registry comment (Tom, 31/07/26). Labeled
explicitly below, not silently reported as "the 3-member scaffold."

Known gap: VM L5+L7 cross-layer ablation is R-condition-only by design (Tom,
01/08/26 — "cleaner signal, no shuffle confound") — no C1 row exists.

Outputs: outputs/analysis/scaffold_ablation/sign_flip_results_summary.csv

Usage:
    uv run python src/stage3_analysis/sign_flip_results.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

CFG = {
    "vm_l5_path":    ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_job7ep_k64.parquet",
    "vm_l7_path":    ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l7_job7ep_k64.parquet",
    "vm_l5l7_path":  ROOT / "outputs/analysis/scaffold_ablation/ablation_cross_l5_l7.parquet",
    "tf_path":       ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_results_long.parquet",
    "tf_dfa_c_path": ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
    "out_path":      ROOT / "outputs/analysis/scaffold_ablation/sign_flip_results_summary.csv",
}


def flip_rate(ablated_correct: pd.Series) -> float:
    return 1 - ablated_correct.mean()


def vm_condition_row(df: pd.DataFrame, target: str, condition: str, label: str, notes: str) -> dict:
    """One VM row. Baseline accuracy comes from this study's own stored full
    logit vector (argmax vs class_id) — self-contained, no external join."""
    sub = df[(df["ablation_target"] == target) & (df["perturbation_condition"] == condition)]
    baseline_correct = sub.apply(lambda r: np.argmax(r["baseline_all_logits"]) == r["class_id"], axis=1)
    ablated_correct = sub["correct_ablated"]
    return {
        "config": label, "condition": condition, "n_clips": len(sub),
        "flip_rate": flip_rate(ablated_correct),
        "baseline_accuracy": baseline_correct.mean(),
        "ablated_accuracy": ablated_correct.mean(),
        "accuracy_drop": baseline_correct.mean() - ablated_correct.mean(),
        "notes": notes,
    }


def vm_cross_layer_row(df: pd.DataFrame) -> dict:
    """L5's 7 + L7's all4 (3 members + near-miss 3347) ablated jointly in one
    forward pass, R only. No full logit vector is stored, but the docstring
    of ablation_cross_l5_l7.py confirms R-correctness is recomputed fresh
    under the dual-spliced baseline — so baseline accuracy is 100% here too."""
    ablated_correct = df["correct_ablated"]
    return {
        "config": "VM_L5+L7_cross", "condition": "R", "n_clips": len(df),
        "flip_rate": flip_rate(ablated_correct),
        "baseline_accuracy": 1.0,
        "ablated_accuracy": ablated_correct.mean(),
        "accuracy_drop": 1.0 - ablated_correct.mean(),
        "notes": "R-only by design (Tom, 01/08 — no shuffle confound); "
                 "L5's 7 + L7's all4 (3 members + near-miss 3347) ablated jointly",
    }


def tf_rows(df: pd.DataFrame, dfa_c: pd.DataFrame) -> list[dict]:
    """TF's TOP12, R and C. run_ablation_tf.py never stores the unablated-C
    "correct" flag or a full logit vector, so C's baseline is joined from
    dfa_mass_delta.parquet's correct_C — verified exact clip_id match against
    this study's own R-eligible pool before trusting the join (see chat log)."""
    top12 = df[df["ablation_target"] == "TOP12"]
    rows = []
    for cond in ("R", "C"):
        sub = top12[top12["perturbation_condition"] == cond]
        ablated_correct = sub["correct_ablated"]
        if cond == "R":
            baseline_acc, note = 1.0, "R-eligibility gate guarantees 100% unablated baseline"
        else:
            merged = sub[["clip_id"]].merge(dfa_c[["clip_id", "correct_C"]], on="clip_id", how="left")
            assert merged["correct_C"].notna().all(), "TF/DFA clip_id join failed for some clips"
            baseline_acc = merged["correct_C"].mean()
            note = "baseline joined from dfa_mass_delta.parquet's correct_C (confirmed matched clip pool)"
        rows.append({
            "config": "TF_TOP12", "condition": cond, "n_clips": len(sub),
            "flip_rate": flip_rate(ablated_correct), "baseline_accuracy": baseline_acc,
            "ablated_accuracy": ablated_correct.mean(),
            "accuracy_drop": baseline_acc - ablated_correct.mean(), "notes": note,
        })
    return rows


def main() -> None:
    vm_l5    = pd.read_parquet(CFG["vm_l5_path"])
    vm_l7    = pd.read_parquet(CFG["vm_l7_path"])
    vm_cross = pd.read_parquet(CFG["vm_l5l7_path"])
    tf       = pd.read_parquet(CFG["tf_path"])
    dfa_c    = pd.read_parquet(CFG["tf_dfa_c_path"])

    l5_note = "7-member gate-derived scaffold (scaffold_selection_consolidated.py)"
    l7_note = "3 gate members (5165/6021/6032) + near-miss 3347, deliberately included"
    rows = [
        vm_condition_row(vm_l5, "all7", "R",  "VM_L5", l5_note),
        vm_condition_row(vm_l5, "all7", "C1", "VM_L5", l5_note),
        vm_condition_row(vm_l7, "all4", "R",  "VM_L7", l7_note),
        vm_condition_row(vm_l7, "all4", "C1", "VM_L7", l7_note),
        vm_cross_layer_row(vm_cross),
        *tf_rows(tf, dfa_c),
    ]

    out = pd.DataFrame(rows)
    out.to_csv(CFG["out_path"], index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
