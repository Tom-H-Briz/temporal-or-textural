"""
Position-lock feature summary.

A feature is position-locked if:
  1. top_tubelet_share_R >= MIN_SHARE_R        — concentrated firing in R
  2. mean_share >= MIN_SHARE_MEAN              — concentration holds across R/C1/A
  3. top_tubelet_idx consistent across R/C1/A  — same tubelet in all conditions
  4. total_abs_R >= MIN_TOTAL_ABS_R            — feature contributes meaningful R mass
  5. top_abs_R >= MIN_TOP_ABS_R               — peak tubelet itself has meaningful mass

Activity filters (4, 5) derived from tubelet_position_lock.parquet for all 32 classes.

Outputs:
  outputs/analysis/dfa_per_tubelet_mass/position_lock_summary.csv
  outputs/analysis/dfa_per_tubelet_mass/position_locked_feature_ids.txt
"""

from pathlib import Path
import pandas as pd

ROOT         = Path(__file__).parent.parent.parent
SCORES_PATH = ROOT / "outputs/analysis/z_position_lock/z_position_lock_scores.csv"
OUT_DIR     = ROOT / "outputs/analysis/z_position_lock"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_MEAN_PER_CLIP_SHARE = 0.90   # avg per-clip dominant-tubelet share in R
MIN_FRAC_MATCHING_MODE  = 0.90   # ≥60% of clips individually peak at the modal tubelet
REQUIRE_POS_CONSISTENT  = True   # modal tubelet agrees across R/C1/A
MIN_TOTAL_ABS_R         = 0.05   # total R DFA mass across all 8 tubelets
MIN_TOP_ABS_R           = 0.02   # peak tubelet R mass



def main() -> None:
    # New CSV is already wide — one row per (class_id, feature_idx), no pivot needed
    merged = pd.read_csv(SCORES_PATH)

    print("\nDistribution of mean_per_clip_share_R:")
    print(merged["mean_per_clip_share_R"].describe(percentiles=[.5,.75,.9,.95,.99]))
    print("\nDistribution of frac_clips_matching_mode_R:")
    print(merged["frac_clips_matching_mode_R"].describe(percentiles=[.5,.75,.9,.95,.99]))
    print()

    mask = (
        (merged["mean_per_clip_share_R"]      >= MIN_MEAN_PER_CLIP_SHARE) &
        (merged["frac_clips_matching_mode_R"] >= MIN_FRAC_MATCHING_MODE)  &
        (merged["total_abs_R"] >= MIN_TOTAL_ABS_R) &
        (merged["top_abs_R"]   >= MIN_TOP_ABS_R)
    )
    # pos_consistent not available in z-only CSV (R condition only)
    if REQUIRE_POS_CONSISTENT and "pos_consistent" in merged.columns:
        mask &= merged["pos_consistent"]

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
    optional  = ["mode_tubelet_C1", "mode_tubelet_A", "pos_consistent",
                 "mean_per_clip_share_C1", "mean_per_clip_share_A",
                 "frac_clips_matching_mode_C1", "frac_clips_matching_mode_A"]
    cols = (
        ["feature_idx", "n_classes_locked", "class_id", "n_clips",
         "mode_tubelet_R", "mean_per_clip_share_R", "frac_clips_matching_mode_R",
         "total_abs_R", "top_abs_R"]
        + [c for c in optional if c in available]
    )
    out_path = OUT_DIR / "position_lock_summary.csv"
    result[cols].to_csv(out_path, index=False)

    unique_ids = sorted(result["feature_idx"].unique())
    # One row per unique feature: feature_idx, locked tubelet (modal top_t_R across classes)
    feat_tubelet = (
        result.groupby("feature_idx")["mode_tubelet_R"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .reset_index()
        .rename(columns={"mode_tubelet_R": "tubelet_idx"})
        .sort_values("feature_idx")
    )
    id_path = OUT_DIR / "position_locked_feature_ids.txt"
    id_path.write_text(
        "feature_idx,tubelet_idx\n" +
        "\n".join(f"{int(r.feature_idx)},{int(r.tubelet_idx)}"
                  for _, r in feat_tubelet.iterrows())
    )

    print(f"  {len(result)} class×feature pairs  |  {len(unique_ids)} unique features / 6144 ({len(unique_ids)/6144*100:.1f}%)")
    print(f"  → {out_path}")
    print(f"  → {id_path}")
    print(f"\nUnique features by class coverage:")
    summary_cols = ["feature_idx", "n_classes_locked", "mode_tubelet_R",
                    "mean_per_clip_share_R", "frac_clips_matching_mode_R", "total_abs_R"]
    print(result.drop_duplicates("feature_idx")[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
