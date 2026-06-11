"""
Chiral analysis: rank temporally-specific features from chiral_extraction.parquet.

Filter per clip: feature f survives if
  r_token_fire_counts[f] >= activity_threshold   (active in real clip)
  a_token_fire_counts[f] <  activity_threshold   (absent in midpoint still)
  b_token_fire_counts[f] <  activity_threshold   (absent in first/last)

Aggregate per direction (L2R = class 93, R2L = class 94):
  n_clips_surviving_filter  — clips where f passes all three conditions
  n_clips_active_R          — clips where f fires in R (regardless of A/B)
  mean_dfa_R                — mean r_per_feature_summary[f] over surviving clips
  consistency_frac          — n_clips_surviving / n_clips_active_R

Full survivor set requires:
  n_clips_surviving_filter >= min_clips_surviving
  mean_dfa_R >= percentile(all_active_r_dfa, min_dfa_percentile)

where all_active_r_dfa is mean_dfa_R computed over R-active clips, for every
feature that fires in R at least once.

Outputs (outputs/analysis/tearing/):
  tearing_features_two_pieces.csv — top_n survivors for class 149
  tearing_features_little_bit.csv — top_n survivors for class 150
  tearing_features_contrast.csv   — full survivor sets, direction-exclusive features only
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "extraction_path": str(ROOT / "outputs/analysis/tearing_extraction.parquet"),
    "activity_threshold": 17,
    "min_clips_surviving": 3,
    "min_dfa_percentile": 50,
    "top_n_features": 50,
    "output_dir": str(ROOT / "outputs/analysis/tearing/"),
    "class_ids": {149: "two_pieces", 150: "little_bit"},
}


def load_arrays(df: pd.DataFrame, class_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-clip tensor columns for one class into (n_clips, dict_size) arrays."""
    rows = df[df["class_id"] == class_id]
    r_pfs = np.stack([np.array(v) for v in rows["r_per_feature_summary"]])   # (n, D)
    r_tfc = np.stack([np.array(v) for v in rows["r_token_fire_counts"]])     # (n, D)
    a_tfc = np.stack([np.array(v) for v in rows["a_token_fire_counts"]])     # (n, D)
    b_tfc = np.stack([np.array(v) for v in rows["b_token_fire_counts"]])     # (n, D)
    return r_pfs, r_tfc, a_tfc, b_tfc


def compute_survivors(
    r_pfs: np.ndarray,
    r_tfc: np.ndarray,
    a_tfc: np.ndarray,
    b_tfc: np.ndarray,
    cfg: dict,
) -> pd.DataFrame:
    """
    Returns full survivor DataFrame sorted by mean_dfa_R descending.
    Percentile threshold is computed over features with n_clips_active_R >= 1
    using mean DFA over R-active clips (broader than surviving-only mean).
    """
    threshold = cfg["activity_threshold"]

    r_active  = r_tfc >= threshold                   # (n, D) bool
    survives  = r_active & (a_tfc < threshold) & (b_tfc < threshold)

    n_active    = r_active.sum(axis=0).astype(float)   # (D,)
    n_surviving = survives.sum(axis=0).astype(float)   # (D,)

    # mean DFA over R-active clips — used only for percentile baseline
    mean_dfa_active = np.where(
        n_active > 0,
        (r_pfs * r_active).sum(axis=0) / np.where(n_active > 0, n_active, 1.0),
        0.0,
    )

    # mean DFA over surviving clips — reported in output
    mean_dfa_surviving = np.where(
        n_surviving > 0,
        (r_pfs * survives).sum(axis=0) / np.where(n_surviving > 0, n_surviving, 1.0),
        0.0,
    )

    # percentile threshold over all features that fire in R at least once
    active_dfa_vals = mean_dfa_active[n_active >= 1]
    if len(active_dfa_vals) == 0:
        raise ValueError("No features active in R — check activity_threshold and data")
    pct_threshold = float(np.percentile(active_dfa_vals, cfg["min_dfa_percentile"]))
    log.info(
        f"  R-active features: {int((n_active >= 1).sum())}  "
        f"  p{cfg['min_dfa_percentile']} DFA threshold: {pct_threshold:.6f}"
    )

    consistency = np.where(n_active > 0, n_surviving / n_active, 0.0)

    survivor_mask = (
        (n_surviving >= cfg["min_clips_surviving"]) &
        (mean_dfa_surviving >= pct_threshold)
    )
    fids = np.where(survivor_mask)[0]
    log.info(f"  Survivors: {len(fids)}")

    return pd.DataFrame({
        "feature_id":               fids,
        "mean_dfa_R":               mean_dfa_surviving[fids],
        "n_clips_surviving_filter": n_surviving[fids].astype(int),
        "n_clips_active_R":         n_active[fids].astype(int),
        "consistency_frac":         consistency[fids],
    }).sort_values("mean_dfa_R", ascending=False).reset_index(drop=True)


def save_csvs(
    l2r: pd.DataFrame,
    r2l: pd.DataFrame,
    cfg: dict,
) -> None:
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    top_n = cfg["top_n_features"]

    l2r.head(top_n).to_csv(out / "tearing_features_two_pieces.csv", index=False)
    log.info(f"Saved → {out / 'tearing_features_two_pieces.csv'}  ({min(len(l2r), top_n)} rows)")

    r2l.head(top_n).to_csv(out / "tearing_features_little_bit.csv", index=False)
    log.info(f"Saved → {out / 'tearing_features_little_bit.csv'}  ({min(len(r2l), top_n)} rows)")

    l2r_ids = set(l2r["feature_id"])
    r2l_ids = set(r2l["feature_id"])

    l2r_only = l2r[l2r["feature_id"].isin(l2r_ids - r2l_ids)].copy()
    l2r_only["direction"] = "two_pieces_only"

    r2l_only = r2l[r2l["feature_id"].isin(r2l_ids - l2r_ids)].copy()
    r2l_only["direction"] = "little_bit_only"

    contrast = pd.concat([l2r_only, r2l_only], ignore_index=True)
    contrast.to_csv(out / "tearing_features_contrast.csv", index=False)
    log.info(
        f"Saved → {out / 'tearing_features_contrast.csv'}  "
        f"({len(l2r_only)} L2R_only, {len(r2l_only)} R2L_only)"
    )


def main() -> None:
    cfg = CFG
    df = pd.read_parquet(cfg["extraction_path"])
    log.info(f"Loaded extraction: {len(df)} rows, classes {sorted(df['class_id'].unique())}")

    results = {}
    for class_id, label in cfg["class_ids"].items():
        log.info(f"Processing class {class_id} ({label})...")
        r_pfs, r_tfc, a_tfc, b_tfc = load_arrays(df, class_id)
        log.info(f"  Clips: {r_pfs.shape[0]}  Dict size: {r_pfs.shape[1]}")
        results[label] = compute_survivors(r_pfs, r_tfc, a_tfc, b_tfc, cfg)

    save_csvs(results["two_pieces"], results["little_bit"], cfg)


if __name__ == "__main__":
    main()
