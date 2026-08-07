"""
VM/TF accuracy gap vs. VM's own L5 position-lock ablation impact — per class.

Fixes a sourcing bug from earlier ad-hoc versions: ablation_l5_vs_l7_class_impact.csv's
"L5" numbers actually came from ablation_results_long_clean7_020726.parquet, target
"clean7" (features 1842/1990/1996/3513/3558/5552/5578) — a superseded candidate
scaffold, NOT the current ablation_targets.py registry entry for ("ssv2", 5)
(features 358/449/917/2093/3516/3938/5004, target "all7", in
ablation_results_long_l5_job7ep_k64.parquet). This reads the correct one directly.

The ablation parquet only contains clips that were already R-correct (upstream
filtering), so its own baseline is trivially 100% — not usable for a real delta.

Bootstrap is clip-PAIRED, not independent: an earlier version resampled VM
accuracy, TF accuracy, and L5-ablated accuracy as three separate binomials,
which overstates uncertainty — VM and TF are scored on the same clips per
class (n_vm == n_tf for every class), and some clips are just easier than
others across both. This resamples clip_id once per draw (from
vm_tf_r_accuracy_per_clip.py's per-clip VM+TF correctness) and recomputes all
three quantities from that same resampled set, so shared per-clip difficulty
cancels out of the deltas instead of inflating their variance.

Outputs (outputs/analysis/scaffold_ablation/):
    vm_tf_accuracy_vs_l5_ablation.csv

Usage:
    uv run python src/stage3_analysis/vm_tf_accuracy_vs_l5_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "analysis" / "scaffold_ablation"
L5_PARQUET = OUT_DIR / "ablation_results_long_l5_job7ep_k64.parquet"
L5_TARGET = "all7"
PER_CLIP_PARQUET = OUT_DIR / "vm_tf_r_accuracy_per_clip.parquet"
N_BOOTSTRAP = 10_000
RNG_SEED = 0
CI_PCT = (5, 95)  # 90% interval


def load_per_clip_table() -> pd.DataFrame:
    """One row per clip: vm_correct, tf_correct (both from the same clip, same
    model runs — vm_tf_r_accuracy_per_clip.py), and l5_ablated_correct — False
    wherever vm_correct is False (ablation can't fix what was already wrong,
    and such clips were never in the ablation run), else looked up from the
    all7-scaffold ablation parquet. Reports and drops the rare VM-correct clip
    that's missing ablation coverage, rather than silently imputing it."""
    clips = pd.read_parquet(PER_CLIP_PARQUET)
    abl = pd.read_parquet(L5_PARQUET, columns=["clip_id", "perturbation_condition",
                                               "ablation_target", "correct_ablated"])
    abl = abl[(abl["perturbation_condition"] == "R") & (abl["ablation_target"] == L5_TARGET)]
    abl = abl[["clip_id", "correct_ablated"]].rename(columns={"correct_ablated": "l5_ablated_correct"})
    abl["clip_id"] = abl["clip_id"].astype(clips["clip_id"].dtype)

    df = clips.merge(abl, on="clip_id", how="left")
    missing = df["vm_correct"] & df["l5_ablated_correct"].isna()
    if missing.any():
        print(f"  WARNING: {missing.sum()} VM-correct clips missing from the L5 ablation "
              f"run — dropping them from the paired population (not imputing)")
        df = df[~missing]
    df["l5_ablated_correct"] = df["l5_ablated_correct"].fillna(False).astype(bool)
    return df


def bootstrap_class_paired(vm_correct: np.ndarray, tf_correct: np.ndarray,
                            l5_correct: np.ndarray, rng: np.random.Generator) -> dict:
    """Resamples CLIP INDICES once per draw and applies the same resampled index
    set to all three arrays — this is what makes it paired: whatever correlation
    exists between VM/TF/L5-ablated correctness on a given clip (shared clip
    difficulty) is preserved in every draw, rather than being averaged away by
    resampling each quantity's binomial independently."""
    n = len(vm_correct)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    p_vm = vm_correct[idx].mean(axis=1)
    p_tf = tf_correct[idx].mean(axis=1)
    p_l5 = l5_correct[idx].mean(axis=1)  # already full-population ablated accuracy

    delta_unablated = p_vm - p_tf
    delta_l5 = p_l5 - p_vm
    with np.errstate(divide="ignore", invalid="ignore"):
        alignment = np.abs(delta_l5) / delta_unablated
    return {"delta_unablated": delta_unablated, "delta_l5": delta_l5, "alignment": alignment}


def summarize_class(draws: dict) -> dict:
    """Median + 90% CI per quantity. unstable = the delta_unablated CI straddles
    zero — i.e. we can't even confidently say VM beats TF on this class, so its
    alignment ratio (which divides by that delta) isn't meaningful. This replaces
    the earlier flat point-estimate cutoff with an uncertainty-derived one."""
    lo, hi = CI_PCT
    du = draws["delta_unablated"]
    out = {"unstable": bool(np.percentile(du, lo) < 0 < np.percentile(du, hi))}
    for name, vals in draws.items():
        finite = vals[np.isfinite(vals)]
        out[f"{name}_median"] = float(np.median(finite))
        out[f"{name}_ci_lo"] = float(np.percentile(finite, lo))
        out[f"{name}_ci_hi"] = float(np.percentile(finite, hi))
    return out


def build_table() -> pd.DataFrame:
    clips = load_per_clip_table()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for class_id, grp in clips.groupby("class_id"):
        vm_correct = grp["vm_correct"].to_numpy()
        tf_correct = grp["tf_correct"].to_numpy()
        l5_correct = grp["l5_ablated_correct"].to_numpy()
        draws = bootstrap_class_paired(vm_correct, tf_correct, l5_correct, rng)
        rows.append({"class_id": int(class_id), "n_clips": len(grp),
                     "vm_accuracy": vm_correct.mean(), "tf_accuracy": tf_correct.mean(),
                     **summarize_class(draws)})
    out = pd.DataFrame(rows)
    templates = pd.read_csv(ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv",
                            usecols=["class_id", "template"])
    return out.merge(templates, on="class_id").sort_values("class_id").reset_index(drop=True)


def main() -> None:
    out = build_table()
    out.to_csv(OUT_DIR / "vm_tf_accuracy_vs_l5_ablation.csv", index=False)
    cols = ["class_id", "template", "vm_accuracy", "tf_accuracy", "n_clips",
            "delta_unablated_median", "delta_l5_median",
            "alignment_median", "alignment_ci_lo", "alignment_ci_hi", "unstable"]
    print(out[cols].to_string(index=False))

    unstable = out[out["unstable"]]
    stable = out[~out["unstable"]]
    print(f"\n{len(unstable)} classes unstable (90% CI of the VM-TF gap straddles zero): "
          f"{unstable['class_id'].tolist()}")
    med = stable["alignment_median"]
    print(f"Stable n={len(stable)}: across-class median={med.median():.3f}  "
          f"mean={med.mean():.3f}  std={med.std():.3f}  "
          f"IQR=[{med.quantile(0.25):.3f}, {med.quantile(0.75):.3f}]")


if __name__ == "__main__":
    main()
