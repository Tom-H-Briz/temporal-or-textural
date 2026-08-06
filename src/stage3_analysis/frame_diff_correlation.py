"""
Frame-difference correlation for mixed increase/decrease features — CC brief
06/08, item 3 (the "5037-style check", generalised).

For mixed features (from purity_analysis.py's raw 1a), tests whether a
clip's increase-vs-decrease direction tracks that clip's own shuffle-
manufactured discontinuity, independent of feature identity. Pure pixel-
space computation — no model forward pass, no SAE, no DFA. Genuinely new;
nothing in the repo computes this yet.

Reuses the exact R/shuffle frame reconstruction each backbone's DFA data was
built from (dfa_mass_delta_vm.py's preprocess_c1 pairing, dfa_mass_delta.py's
preprocess_c full-permutation) so "shuffle discontinuity" here matches what
produced signed_shuffle, not a fresh reimplementation.

Outputs (outputs/analysis/shuffle_reduction_composition/):
    {backbone}_frame_diff_correlation.csv — feature_id, clip_id, direction, real_frame_diff, shuffle_frame_diff

Usage:
    uv run python src/stage3_analysis/frame_diff_correlation.py
"""

import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))

from perturbation import apply_shuffle
from ToT_utils import FRAME_SAMPLERS, _deterministic_seed

OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"
VIDEO_DIR = ROOT / "data/ssv2/20bn-something-something-v2"
NUM_FRAMES = {"vm": 16, "tf": 8}


def load_raw_frames(clip_id: str) -> list:
    container = av.open(str(VIDEO_DIR / f"{clip_id}.webm"))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def real_and_shuffle_frames(backbone: str, clip_id: str) -> tuple[list, list]:
    """Reconstructs R and C1/C exactly as each backbone's DFA parquet was built:
    VM pairs sampled frames then shuffles the pairs as units (dfa_mass_delta_vm.py
    preprocess_c1); TF shuffles raw frames fully then linspace-samples (dfa_mass_delta.py
    preprocess_c) — same asymmetry already established for run_ablation_tf.py."""
    raw = load_raw_frames(clip_id)
    n = len(raw)
    num_frames = NUM_FRAMES[backbone]
    if backbone == "vm":
        idx = FRAME_SAMPLERS["ssv2"](n, num_frames)
        real = [raw[i] for i in idx]
        pairs = [(real[i], real[i + 1]) for i in range(0, num_frames, 2)]
        order = np.random.default_rng(_deterministic_seed(clip_id)).permutation(len(pairs)).tolist()
        shuffle = [f for i in order for f in pairs[i]]
    else:
        real_idx = FRAME_SAMPLERS["ssv2"](n, num_frames)
        real = [raw[i] for i in real_idx]
        shuffled_raw = apply_shuffle(raw, int(clip_id) % 2**32)
        shuffle_idx = np.linspace(0, n - 1, num_frames).astype(int).tolist()
        shuffle = [shuffled_raw[i] for i in shuffle_idx]
    return real, shuffle


def mean_frame_diff(frames: list) -> float:
    """Mean absolute frame-to-frame pixel difference, averaged over all
    consecutive pairs and pixels — a single scalar discontinuity score per clip."""
    diffs = [np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32)).mean()
             for i in range(len(frames) - 1)]
    return float(np.mean(diffs))


def mixed_instances(backbone: str) -> pd.DataFrame:
    """Per-clip increase/decrease instances for mixed (class,feature) combos only
    — joins purity_analysis.py's raw 1a labels back onto the per-instance detail."""
    detail = pd.read_parquet(OUT_DIR / f"ssv2_{backbone}_full_detail.parquet")
    pooled = pd.read_csv(OUT_DIR / f"{backbone}_purity_pooled.csv")
    mixed = pooled[pooled["label"] == "mixed"][["class_id", "feature_id"]]
    incdec = detail[detail["bucket"].isin(["increase", "decrease"])]
    joined = incdec.merge(mixed, on=["class_id", "feature_id"])
    return joined[["feature_id", "clip_id", "bucket"]].rename(columns={"bucket": "direction"})


def attach_frame_diffs(backbone: str, instances: pd.DataFrame) -> pd.DataFrame:
    cache: dict[str, tuple[float, float]] = {}
    real_diffs, shuffle_diffs = [], []
    for i, clip_id in enumerate(instances["clip_id"]):
        if clip_id not in cache:
            real, shuffle = real_and_shuffle_frames(backbone, str(clip_id))
            cache[clip_id] = (mean_frame_diff(real), mean_frame_diff(shuffle))
        r, s = cache[clip_id]
        real_diffs.append(r)
        shuffle_diffs.append(s)
        if (i + 1) % 500 == 0:
            print(f"  [{backbone}] {i+1}/{len(instances)} instances, {len(cache)} unique clips decoded")
    out = instances.copy()
    out["real_frame_diff"], out["shuffle_frame_diff"] = real_diffs, shuffle_diffs
    return out


def correlate_within_feature(df: pd.DataFrame, backbone: str) -> None:
    """Per mixed feature, correlate (direction==increase) against
    (shuffle_frame_diff - real_frame_diff) across that feature's own clips."""
    df = df.copy()
    df["is_increase"] = (df["direction"] == "increase").astype(int)
    df["diff_gap"] = df["shuffle_frame_diff"] - df["real_frame_diff"]
    rhos = []
    for fid, grp in df.groupby("feature_id"):
        if grp["is_increase"].nunique() < 2 or len(grp) < 4:
            continue
        rho = stats.pointbiserialr(grp["is_increase"], grp["diff_gap"]).statistic
        rhos.append(rho)
    if rhos:
        print(f"  [{backbone}] within-feature direction~diff_gap correlation: "
              f"n_features={len(rhos)} mean_rho={np.mean(rhos):+.3f} median_rho={np.median(rhos):+.3f}")
    else:
        print(f"  [{backbone}] no mixed features had enough variation to correlate")


def main() -> None:
    for backbone in ("vm", "tf"):
        print(f"Processing {backbone}...")
        instances = mixed_instances(backbone)
        df = attach_frame_diffs(backbone, instances)
        df.to_csv(OUT_DIR / f"{backbone}_frame_diff_correlation.csv", index=False)
        correlate_within_feature(df, backbone)


if __name__ == "__main__":
    main()
