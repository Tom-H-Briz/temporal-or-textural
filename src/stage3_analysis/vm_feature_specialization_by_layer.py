"""
Does VM's feature specialization increase monotonically with layer depth?

Two complementary tests, computed identically at L3/L5/L7/L9 (all already-
extracted DFA parquets, same 35-class population, same 6144-dict — pure
re-aggregation, no new extraction):

1. n_classes_present per feature — how many distinct classes a feature
   shows up in (top-10 by |signed_R|, across all clips). Specialized
   features span few classes; generic "motion part" features span many.
2. Per class, the single best feature's within-class coverage (frac of
   that class's clips it appears in) — same metric used for the covering/
   uncovering investigation, now run across every class per layer.

Reuses top10_detail() unchanged (single source of truth for the bucket/
per-instance math), just generalized across layers instead of L7-only.

Outputs (outputs/analysis/shuffle_reduction_composition/):
    vm_feature_specialization_by_layer.csv       — per-layer summary
    vm_best_feature_coverage_by_layer_class.csv  — per (layer, class) best-feature coverage

Usage:
    uv run python src/stage3_analysis/vm_feature_specialization_by_layer.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.clip_shuffle_disruption import top10_detail

OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"
MASS_DELTA_DIR = ROOT / "outputs/analysis/dfa_mass_delta_vm_c1"
DATASETS = {
    "ssv2": {
        3: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_ssv2_l3_job7ep_k64.parquet",
        5: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_l5_job7ep_k64.parquet",
        7: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_l7_job7ep_k64.parquet",
        9: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_l9_job64_k64.parquet",
    },
    "kinetics400": {
        # No L9 extraction exists for K400 — 3 layers only, not silently padded.
        3: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_kinetics400_l3_job7ep_k64.parquet",
        5: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_kinetics400_l5_job7ep_k64.parquet",
        7: MASS_DELTA_DIR / "dfa_mass_delta_vm_c1_kinetics400_l7_job7ep_k64.parquet",
    },
}


def build_layer_detail(dataset: str, layer: int) -> pd.DataFrame:
    """One row per (clip, top-10 feature) for this layer — reuses top10_detail
    unchanged, generalized across layers instead of clip_shuffle_disruption.py's
    L7-only CONFIGS."""
    dfa_df = pd.read_parquet(DATASETS[dataset][layer])
    rows = []
    for _, clip in dfa_df.iterrows():
        r = np.asarray(clip["signed_vec_R"], dtype=np.float32)
        c = np.asarray(clip["signed_vec_C1"], dtype=np.float32)
        detail = top10_detail(r, c)
        detail["clip_id"] = clip["clip_id"]
        detail["class_id"] = clip["class_id"]
        rows.append(detail)
    out = pd.concat(rows, ignore_index=True).rename(columns={"feature_idx": "feature_id"})
    print(f"  L{layer}: {dfa_df['class_id'].nunique()} classes, {len(dfa_df)} clips, {len(out)} instance rows")
    return out


def n_classes_present_stats(detail: pd.DataFrame) -> dict:
    """Per feature, how many distinct classes it appears in — low = specialized,
    high = generic/shared across many classes' top-10."""
    n_classes = detail.groupby("feature_id")["class_id"].nunique()
    return {
        "n_unique_features": len(n_classes),
        "median_n_classes_present": float(n_classes.median()),
        "mean_n_classes_present": float(n_classes.mean()),
        "frac_single_class_features": float((n_classes == 1).mean()),
    }


def best_feature_coverage_per_class(detail: pd.DataFrame, layer: int) -> pd.DataFrame:
    """Per class, the single best feature's within-class coverage — same metric
    used for the covering/uncovering investigation, now for every class."""
    n_clips = detail.groupby("class_id")["clip_id"].nunique()
    coverage = detail.groupby(["class_id", "feature_id"])["clip_id"].nunique().reset_index(name="n_clips_present")
    coverage["frac_of_class_clips"] = coverage["n_clips_present"] / coverage["class_id"].map(n_clips)
    best = coverage.sort_values("frac_of_class_clips", ascending=False).drop_duplicates("class_id")
    best["layer"] = layer
    return best.sort_values("class_id")[["layer", "class_id", "feature_id", "n_clips_present", "frac_of_class_clips"]]


def run_dataset(dataset: str) -> None:
    summary_rows, best_feature_rows = [], []
    for layer in sorted(DATASETS[dataset]):
        print(f"Processing {dataset} L{layer}...")
        detail = build_layer_detail(dataset, layer)
        stats = n_classes_present_stats(detail)
        stats["layer"] = layer

        best = best_feature_coverage_per_class(detail, layer)
        stats["mean_best_feature_coverage"] = float(best["frac_of_class_clips"].mean())
        stats["median_best_feature_coverage"] = float(best["frac_of_class_clips"].median())
        summary_rows.append(stats)
        best_feature_rows.append(best)

    summary = pd.DataFrame(summary_rows)[["layer", "n_unique_features", "median_n_classes_present",
                                          "mean_n_classes_present", "frac_single_class_features",
                                          "mean_best_feature_coverage", "median_best_feature_coverage"]]
    summary.to_csv(OUT_DIR / f"vm_feature_specialization_by_layer_{dataset}.csv", index=False)
    pd.concat(best_feature_rows, ignore_index=True).to_csv(
        OUT_DIR / f"vm_best_feature_coverage_by_layer_class_{dataset}.csv", index=False)

    print(f"\n=== {dataset} ===")
    print(summary.to_string(index=False))


def main() -> None:
    for dataset in DATASETS:
        run_dataset(dataset)


if __name__ == "__main__":
    main()
