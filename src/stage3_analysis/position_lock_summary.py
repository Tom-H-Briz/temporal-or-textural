"""
Position-lock feature summary.

A feature is position-locked if:
  1. top_tubelet_share_R >= MIN_SHARE_R        — concentrated firing in R
  2. mean_share >= MIN_SHARE_MEAN              — concentration holds across R/C1/A
  3. top_tubelet_idx consistent across R/C1/A  — same tubelet in all conditions
  4. total_abs_R >= MIN_TOTAL_ABS_R            — feature contributes meaningful R mass
  5. top_abs_R >= MIN_TOP_ABS_R               — peak tubelet itself has meaningful mass

Criterion uses the raw-activation (z-based) columns, not the DFA-based ones —
matches this script's original behavior (it always read z_position_lock_scores.csv).

Since position_lock_extraction.py (31/07 merge) writes one combined CSV per
(model, layer, dataset, job, k) config, this script now takes that CSV's path
as an argument rather than a single hardcoded file.

Outputs (alongside the input CSV, same suffix):
  outputs/analysis/position_lock/position_lock_summary_{suffix}.csv
  outputs/analysis/position_lock/position_locked_feature_ids_{suffix}.txt
"""

import argparse
import re
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_MEAN_PER_CLIP_SHARE = 0.90   # avg per-clip dominant-tubelet share in R
MIN_FRAC_MATCHING_MODE  = 0.90   # ≥60% of clips individually peak at the modal tubelet
REQUIRE_POS_CONSISTENT  = True   # modal tubelet agrees across R/C1/A
MIN_TOTAL_ABS_R         = 0.05   # total R DFA mass across all 8 tubelets
MIN_TOP_ABS_R           = 0.02   # peak tubelet R mass



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_csv", type=Path,
                        help="path to a position_lock_scores_{suffix}.csv from position_lock_extraction.py")
    args = parser.parse_args()

    out_dir = args.scores_csv.parent
    suffix_match = re.match(r"position_lock_scores_(.+)\.csv$", args.scores_csv.name)
    suffix = suffix_match.group(1) if suffix_match else args.scores_csv.stem

    # New CSV is already wide — one row per (class_id, feature_idx), no pivot needed
    merged = pd.read_csv(args.scores_csv)

    print("\nDistribution of mean_per_clip_share_z_R:")
    print(merged["mean_per_clip_share_z_R"].describe(percentiles=[.5,.75,.9,.95,.99]))
    print("\nDistribution of frac_clips_matching_mode_z_R:")
    print(merged["frac_clips_matching_mode_z_R"].describe(percentiles=[.5,.75,.9,.95,.99]))
    print()

    mask = (
        (merged["mean_per_clip_share_z_R"]      >= MIN_MEAN_PER_CLIP_SHARE) &
        (merged["frac_clips_matching_mode_z_R"] >= MIN_FRAC_MATCHING_MODE)  &
        (merged["total_abs_z_R"] >= MIN_TOTAL_ABS_R) &
        (merged["top_abs_z_R"]   >= MIN_TOP_ABS_R)
    )
    if REQUIRE_POS_CONSISTENT and "pos_consistent_z" in merged.columns:
        mask &= merged["pos_consistent_z"]

    result = merged[mask].copy()

    # Count how many classes each feature locks in
    class_counts = result.groupby("feature_idx")["class_id"].count().rename("n_classes_locked")
    result = result.merge(class_counts, on="feature_idx")

    # Sort by n_classes_locked desc, then feature_idx, then class_id
    result = result.sort_values(
        ["n_classes_locked", "feature_idx", "class_id"],
        ascending=[False, True, True]
    ).reset_index(drop=True)

    available = set(result.columns)
    optional  = ["mode_tubelet_z_C1", "mode_tubelet_z_A", "pos_consistent_z",
                 "mean_per_clip_share_z_C1", "mean_per_clip_share_z_A",
                 "frac_clips_matching_mode_z_C1", "frac_clips_matching_mode_z_A"]
    cols = (
        ["feature_idx", "n_classes_locked", "class_id", "n_clips_z",
         "mode_tubelet_z_R", "mean_per_clip_share_z_R", "frac_clips_matching_mode_z_R",
         "total_abs_z_R", "top_abs_z_R"]
        + [c for c in optional if c in available]
    )
    out_path = out_dir / f"position_lock_summary_{suffix}.csv"
    result[cols].to_csv(out_path, index=False)

    unique_ids = sorted(result["feature_idx"].unique())
    # One row per unique feature: feature_idx, locked tubelet (modal top_t_R across classes)
    feat_tubelet = (
        result.groupby("feature_idx")["mode_tubelet_z_R"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .reset_index()
        .rename(columns={"mode_tubelet_z_R": "tubelet_idx"})
        .sort_values("feature_idx")
    )
    id_path = out_dir / f"position_locked_feature_ids_{suffix}.txt"
    id_path.write_text(
        "feature_idx,tubelet_idx\n" +
        "\n".join(f"{int(r.feature_idx)},{int(r.tubelet_idx)}"
                  for _, r in feat_tubelet.iterrows())
    )

    print(f"  {len(result)} class×feature pairs  |  {len(unique_ids)} unique features / 6144 ({len(unique_ids)/6144*100:.1f}%)")
    print(f"  → {out_path}")
    print(f"  → {id_path}")
    print(f"\nUnique features by class coverage:")
    summary_cols = ["feature_idx", "n_classes_locked", "mode_tubelet_z_R",
                    "mean_per_clip_share_z_R", "frac_clips_matching_mode_z_R", "total_abs_z_R"]
    print(result.drop_duplicates("feature_idx")[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
