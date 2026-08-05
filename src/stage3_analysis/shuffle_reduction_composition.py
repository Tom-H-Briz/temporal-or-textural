"""
Shuffle-reduction composition vs behavioural loss — CC brief 05/08, addendum same day.

Per eligible class (R-accuracy >= 40%, independent per config): take the
top-10 features by mean_abs_R, split each into decrease/same/increase by
relative_change under shuffle (5% band), and compare the three composition
fractions against relative accuracy loss. Three-way replaces the original
binary reducer/non-reducer split — the binary "non-reducer" bucket silently
pooled genuinely order-invariant features with features that spike upward
from shuffle-manufactured motion artifacts (see addendum: feature 5037),
which is suspected of running the original correlation backwards. Three
configs, kept separate — no pooling (pooling would conflate mechanism with
between-config baseline differences).

Usage:
    uv run python src/stage3_analysis/shuffle_reduction_composition.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent

ELIGIBILITY_THRESHOLD = 0.40
TOP_N = 10
CHANGE_BAND = 0.05  # |relative_change| <= this -> "same" bucket
OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"

# Post-fix (post-30/07) sources only — see Step 0 audit, 05/08.
CONFIGS = {
    "ssv2_vm": dict(
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_l7_job7ep_k64.parquet",
        shuffle_col="signed_vec_C1",
        r_acc_csv=ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv",
        shuffle_acc_csv=ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_C1.csv",
    ),
    "ssv2_tf": dict(
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
        shuffle_col="signed_vec_C",
        r_acc_csv=ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv",
        shuffle_acc_csv=ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF_C.csv",
    ),
    "k400_vm": dict(
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_kinetics400_l7_job7ep_k64.parquet",
        shuffle_col="signed_vec_C1",
        r_acc_csv=ROOT / "outputs/stage1_class_selection_VM_kinetics/per_class_accuracy_VM_kinetics_R.csv",
        shuffle_acc_csv=ROOT / "outputs/stage1_class_selection_VM_kinetics/per_class_accuracy_VM_kinetics_C1.csv",
    ),
}


def top10_buckets(clips: pd.DataFrame, shuffle_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-10 feature_idx by mean_abs_R, their R-mass, and decrease/same/increase bucket labels."""
    s_r = np.stack(clips["signed_vec_R"].to_numpy()).astype(np.float32)
    s_c = np.stack(clips[shuffle_col].to_numpy()).astype(np.float32)
    mean_abs_r = np.abs(s_r).mean(axis=0)
    mean_abs_c = np.abs(s_c).mean(axis=0)
    top10 = np.argsort(-mean_abs_r)[:TOP_N]
    rel_change = (mean_abs_c[top10] - mean_abs_r[top10]) / mean_abs_r[top10]
    bucket = np.where(rel_change < -CHANGE_BAND, "decrease",
              np.where(rel_change > CHANGE_BAND, "increase", "same"))
    return top10, mean_abs_r[top10], bucket


def compute_class_row(clips: pd.DataFrame, shuffle_col: str, class_id: int,
                       r_acc: float, shuffle_acc: float) -> dict:
    _, top10_r_mass, bucket = top10_buckets(clips, shuffle_col)
    total = top10_r_mass.sum()
    counts = {b: int((bucket == b).sum()) for b in ("decrease", "same", "increase")}
    fracs = {b: float(top10_r_mass[bucket == b].sum() / total) for b in ("decrease", "same", "increase")}
    return {
        "class_id": class_id,
        "n_clips": len(clips),
        "R_acc": r_acc,
        "shuffle_acc": shuffle_acc,
        "rel_acc_loss": (r_acc - shuffle_acc) / r_acc,
        "n_decrease_of_10": counts["decrease"],
        "n_same_of_10": counts["same"],
        "n_increase_of_10": counts["increase"],
        "frac_decrease": fracs["decrease"],
        "frac_same": fracs["same"],
        "frac_increase": fracs["increase"],
    }


def eligible_classes(dfa_df: pd.DataFrame, r_acc_df: pd.DataFrame) -> list[int]:
    passed = set(r_acc_df.loc[r_acc_df["accuracy"] >= ELIGIBILITY_THRESHOLD, "class_id"])
    present = set(dfa_df["class_id"].unique())
    return sorted(passed & present)


def build_config_table(name: str, cfg: dict) -> pd.DataFrame:
    dfa_df = pd.read_parquet(cfg["dfa_parquet"])
    r_acc_df = pd.read_csv(cfg["r_acc_csv"])
    shuffle_acc_df = pd.read_csv(cfg["shuffle_acc_csv"]).set_index("class_id")["accuracy"]
    classes = eligible_classes(dfa_df, r_acc_df)
    r_acc_map = r_acc_df.set_index("class_id")["accuracy"]

    rows = []
    for cid in classes:
        if cid not in shuffle_acc_df.index:
            print(f"  [{name}] WARNING: class {cid} eligible but missing from shuffle accuracy — skipped")
            continue
        clips = dfa_df[dfa_df["class_id"] == cid]
        rows.append(compute_class_row(clips, cfg["shuffle_col"], cid,
                                       float(r_acc_map[cid]), float(shuffle_acc_df[cid])))
    return pd.DataFrame(rows)


def report_correlations(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] N={len(df)} eligible classes")
    for frac_col in ("frac_decrease", "frac_increase", "frac_same"):
        pear = stats.pearsonr(df[frac_col], df["rel_acc_loss"])
        spear = stats.spearmanr(df[frac_col], df["rel_acc_loss"])
        print(f"  {frac_col} vs rel_acc_loss:  Pearson r={pear.statistic:.3f} (p={pear.pvalue:.4f})  "
              f"Spearman rho={spear.statistic:.3f} (p={spear.pvalue:.4f})")


def report_cross_config(tables: dict[str, pd.DataFrame]) -> None:
    print("\n=== Cross-config comparison (descriptive, not pooled) ===")
    for name, df in tables.items():
        print(f"  {name}: frac_decrease mean={df['frac_decrease'].mean():.3f} "
              f"frac_increase mean={df['frac_increase'].mean():.3f} "
              f"frac_same mean={df['frac_same'].mean():.3f} | "
              f"rel_acc_loss mean={df['rel_acc_loss'].mean():.3f} median={df['rel_acc_loss'].median():.3f}")

    vm, tf = tables["ssv2_vm"], tables["ssv2_tf"]
    shared = sorted(set(vm["class_id"]) & set(tf["class_id"]))
    gap = (vm.set_index("class_id").loc[shared, "R_acc"]
           - tf.set_index("class_id").loc[shared, "R_acc"]).abs()
    print(f"\n  Covariate note — VM vs TF SSv2 R-accuracy gap, shared eligible classes (n={len(shared)}):")
    print(f"    median |VM_R_acc - TF_R_acc| = {gap.median():.4f}  "
          f"(comparison is NOT accuracy-matched — context only, see brief)")


CSV_COLUMNS = ["class_id", "n_clips", "R_acc", "shuffle_acc", "rel_acc_loss",
               "n_decrease_of_10", "n_same_of_10", "n_increase_of_10",
               "frac_decrease", "frac_same", "frac_increase"]

SPOT_CHECK_FEATURE = 6135  # cover/uncover set, 06/07 — expected in "decrease" bucket
SPOT_CHECK_CLASSES = [6, 171]  # TF, where feature 6135 was logged


def spot_check(dfa_df: pd.DataFrame, shuffle_col: str) -> None:
    print(f"\n=== Spot-check: feature {SPOT_CHECK_FEATURE} in TF classes {SPOT_CHECK_CLASSES} ===")
    for cid in SPOT_CHECK_CLASSES:
        clips = dfa_df[dfa_df["class_id"] == cid]
        top10, _, bucket = top10_buckets(clips, shuffle_col)
        if SPOT_CHECK_FEATURE not in top10:
            print(f"  class {cid}: feature {SPOT_CHECK_FEATURE} NOT in top-10 — not comparable this pass")
            continue
        b = bucket[list(top10).index(SPOT_CHECK_FEATURE)]
        flag = "" if b == "decrease" else "  <-- MISMATCH, expected decrease"
        print(f"  class {cid}: feature {SPOT_CHECK_FEATURE} -> bucket '{b}'{flag}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, cfg in CONFIGS.items():
        print(f"Processing {name}...")
        df = build_config_table(name, cfg)
        tables[name] = df
        df[CSV_COLUMNS].to_csv(OUT_DIR / f"{name}_shuffle_composition.csv", index=False)
        report_correlations(name, df)
    report_cross_config(tables)
    spot_check(pd.read_parquet(CONFIGS["ssv2_tf"]["dfa_parquet"]), str(CONFIGS["ssv2_tf"]["shuffle_col"]))


if __name__ == "__main__":
    main()
