"""
Per-class feature delta ranking — signed DFA mass shift under shuffle.

Reads dfa_mass_delta.parquet (full signed vectors). No DFA engine calls.

Outputs per class:
    outputs/analysis/per_class_feature_delta/class_{id}_feature_ranking.csv
    outputs/analysis/per_class_feature_delta/class_{id}_dfa_histogram.png

Plus:
    outputs/analysis/per_class_feature_delta/decile_thresholds.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent
PARQUET = ROOT / "outputs" / "analysis" / "dfa_mass_delta" / "dfa_mass_delta.parquet"
OUT_DIR = ROOT / "outputs" / "analysis" / "per_class_feature_delta"

CLASSES = [6, 30, 59, 164, 169, 171, 173]


def compute_feature_stats(clips: pd.DataFrame) -> pd.DataFrame:
    s_r = np.stack([np.asarray(v) for v in clips["signed_vec_R"]]).astype(np.float32)
    s_c = np.stack([np.asarray(v) for v in clips["signed_vec_C"]]).astype(np.float32)
    s_a = np.stack([np.asarray(v) for v in clips["signed_vec_A"]]).astype(np.float32)
    mean_s_R   = s_r.mean(axis=0)
    mean_s_C   = s_c.mean(axis=0)
    mean_s_A   = s_a.mean(axis=0)
    mean_abs_R = np.abs(s_r).mean(axis=0)
    mean_abs_C = np.abs(s_c).mean(axis=0)
    mean_abs_A = np.abs(s_a).mean(axis=0)
    sign_R     = np.where((s_r > 1e-8).sum(axis=0) >= (s_r < -1e-8).sum(axis=0), 1, -1)
    return pd.DataFrame({
        "feature_idx":  np.arange(s_r.shape[1]),
        "mean_s_R":     mean_s_R,
        "mean_s_C":     mean_s_C,
        "mean_s_A":     mean_s_A,
        "delta":        mean_s_R - mean_s_C,
        "delta_A":      mean_s_R - mean_s_A,
        "mean_abs_R":   mean_abs_R,
        "mean_abs_C":   mean_abs_C,
        "mean_abs_A":   mean_abs_A,
        "abs_delta":    mean_abs_R - mean_abs_C,
        "abs_delta_A":  mean_abs_R - mean_abs_A,
        "flip_rate":    (np.sign(s_r) != np.sign(s_c)).mean(axis=0),
        "flip_rate_A":  (np.sign(s_r) != np.sign(s_a)).mean(axis=0),
        "sign_R":       sign_R,
        "pct_active_R": (np.abs(s_r) > 1e-8).mean(axis=0),
    })


def assign_deciles(df: pd.DataFrame) -> pd.DataFrame:
    bucket = 6144 // 10  # 614; N90 absorbs the remainder
    ranked = df.sort_values("mean_abs_R", ascending=False).reset_index(drop=True)
    labels = []
    for i in range(1, 10):
        labels.extend([f"N{i * 10}"] * bucket)
    labels.extend(["N100"] * (6144 - 9 * bucket))
    ranked["dfa_decile"] = labels
    return df.merge(ranked[["feature_idx", "dfa_decile"]], on="feature_idx")


def make_histogram(df: pd.DataFrame, class_id: int, n_clips: int, out_dir: Path) -> None:
    sdf   = df.sort_values("mean_abs_R", ascending=False).reset_index(drop=True)
    bsize = 6144 // 10
    y_top = sdf["mean_abs_R"].max()
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, 6145)
    ax.plot(x, sdf["mean_abs_R"].values, linewidth=0.8, label="R")
    ax.plot(x, sdf["mean_abs_C"].values, linewidth=0.8, linestyle="--", label="C")
    ax.plot(x, sdf["mean_abs_A"].values, linewidth=0.8, linestyle=":", label="A")
    ax.set_yscale("log")
    for i in range(1, 10):
        x = i * bsize
        ax.axvline(x, color="grey", linewidth=0.6, linestyle="--")
        ax.text(x - bsize / 2, y_top, f"N{i * 10}", ha="center", va="top", fontsize=8)
    ax.text((9 * bsize + 6144) / 2, y_top, "N100", ha="center", va="top", fontsize=8)
    ax.legend(fontsize=8)
    ax.set_xlabel("feature rank (sorted by mean_abs_R descending)")
    ax.set_ylabel("mean_abs (log scale)")
    ax.set_title(f"Class {class_id} — DFA mass distribution (n={n_clips} clips)")
    fig.tight_layout()
    fig.savefig(out_dir / f"class_{class_id}_dfa_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Histogram → class_{class_id}_dfa_histogram.png")


def process_class(class_id: int, all_clips: pd.DataFrame, out_dir: Path) -> dict:
    clips   = all_clips[all_clips["class_id"] == class_id].copy()
    n_clips = len(clips)
    print(f"  Class {class_id}: {n_clips} clips")
    df = compute_feature_stats(clips)
    df = assign_deciles(df)
    df = df.sort_values("delta", ascending=False).reset_index(drop=True)
    df["rank"]           = np.arange(1, len(df) + 1)
    df["delta_rank_pct"] = df["rank"] / 6144 * 100
    df["n_clips"]        = n_clips
    cols = ["feature_idx", "rank", "delta_rank_pct", "dfa_decile",
            "mean_s_R", "mean_s_C", "mean_s_A", "delta", "delta_A",
            "mean_abs_R", "mean_abs_C", "mean_abs_A", "abs_delta", "abs_delta_A",
            "flip_rate", "flip_rate_A", "sign_R", "pct_active_R", "n_clips"]
    df[cols].to_csv(out_dir / f"class_{class_id}_feature_ranking.csv", index=False)
    make_histogram(df, class_id, n_clips, out_dir)
    row = {"class_id": class_id, "n_clips": n_clips}
    for label in [f"N{i * 10}" for i in range(1, 10)] + ["N100"]:
        row[f"{label}_threshold"] = df[df["dfa_decile"] == label]["mean_abs_R"].min()
    return row


def main() -> None:
    all_clips = pd.read_parquet(PARQUET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_rows = []
    for class_id in CLASSES:
        print(f"Processing class {class_id}...")
        threshold_rows.append(process_class(class_id, all_clips, OUT_DIR))
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "decile_thresholds.csv", index=False)
    print(f"  Thresholds → {OUT_DIR / 'decile_thresholds.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
