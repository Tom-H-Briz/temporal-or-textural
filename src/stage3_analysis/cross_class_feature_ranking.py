"""
Cross-class feature ranking — temporal specificity across SL classes.

Pivots per-class delta_rank_pct into a feature × class table, then computes
temporal_static_gap = mean_static_rank_pct − mean_temporal_rank_pct.
Higher gap = feature delta more concentrated in temporal classes.

Features filtered to top-20% mean_abs_R in at least one class before pivoting.

Usage:
    uv run python src/stage3_analysis/cross_class_feature_ranking.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

#BACKBONE  = "vm"
#CONDITION = "c1"

# --- TF equivalent ---
BACKBONE  = "tf"
CONDITION = "c"
IN_DIR  = ROOT / "outputs" / "analysis" / "per_class_feature_delta"
OUT_DIR = ROOT / "outputs" / "analysis" / "cross_class_feature_ranking_tf_c"

TEMPORAL_CLASSES = [6, 30, 164, 171]
STATIC_CLASSES   = [59, 169, 173]

#IN_DIR  = ROOT / "outputs" / "analysis" / f"per_class_feature_delta_{BACKBONE}_{CONDITION}"
#OUT_DIR = ROOT / "outputs" / "analysis" / f"cross_class_feature_ranking_{BACKBONE}_{CONDITION}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CLASSES = TEMPORAL_CLASSES + STATIC_CLASSES


def load_per_class(in_dir: Path, classes: list[int]) -> dict[int, pd.DataFrame]:
    """Load per-class CSVs, indexed by feature_idx."""
    result = {}
    for cid in classes:
        path = in_dir / f"class_{cid}_feature_ranking.csv"
        df   = pd.read_csv(path, usecols=["feature_idx", "delta_rank_pct", "mean_abs_R"])
        result[cid] = df.set_index("feature_idx")
    return result


def filter_active(per_class: dict[int, pd.DataFrame]) -> list[int]:
    """Keep features in top-20% mean_abs_R in at least one class."""
    thresholds   = {cid: np.percentile(df["mean_abs_R"], 80) for cid, df in per_class.items()}
    all_features = next(iter(per_class.values())).index.tolist()
    active = [
        f for f in all_features
        if any(per_class[cid].loc[f, "mean_abs_R"] > thresholds[cid] for cid in per_class)
    ]
    print(f"  Active features: {len(active)} / {len(all_features)} (top-20% mean_abs_R in ≥1 class)")
    return active


def pivot(per_class: dict[int, pd.DataFrame], active: list[int], classes: list[int]) -> pd.DataFrame:
    frames = []
    for cid in classes:
        df = per_class[cid].loc[active].rename(columns={
            "delta_rank_pct": f"delta_rank_pct_{cid}",
            "mean_abs_R":     f"mean_abs_R_{cid}",
        })
        frames.append(df)
    return pd.concat(frames, axis=1).reset_index()


def main() -> None:
    per_class = load_per_class(IN_DIR, ALL_CLASSES)
    active    = filter_active(per_class)
    df        = pivot(per_class, active, ALL_CLASSES)

    temporal_cols = [f"delta_rank_pct_{c}" for c in TEMPORAL_CLASSES]
    static_cols   = [f"delta_rank_pct_{c}" for c in STATIC_CLASSES]

    df["mean_temporal_rank_pct"] = df[temporal_cols].mean(axis=1)
    df["mean_static_rank_pct"]   = df[static_cols].mean(axis=1)
    df["temporal_static_gap"]    = df["mean_static_rank_pct"] - df["mean_temporal_rank_pct"]

    interleaved = [col for c in ALL_CLASSES
                   for col in (f"delta_rank_pct_{c}", f"mean_abs_R_{c}")]
    out_cols = (["feature_idx"] + interleaved +
                ["mean_temporal_rank_pct", "mean_static_rank_pct", "temporal_static_gap"])
    df = df[out_cols].sort_values("temporal_static_gap", ascending=False).reset_index(drop=True)

    out_path = OUT_DIR / f"cross_class_feature_ranking_{BACKBONE}_{CONDITION}.csv"
    with open(out_path, "w") as f:
        f.write(f"# active_features={len(active)} / 6144 (top-20% mean_abs_R in at least one class)\n")
        df.to_csv(f, index=False)

    print(f"  {len(df)} features → {out_path}")
    print(f"  Gap range: [{df['temporal_static_gap'].min():.2f}, {df['temporal_static_gap'].max():.2f}]")
    print(f"  Top 5 by gap:")
    print(df.head(5)[["feature_idx", "mean_temporal_rank_pct", "mean_static_rank_pct", "temporal_static_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
