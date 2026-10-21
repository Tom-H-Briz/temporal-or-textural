"""Per-clip profile of one L9 SAE feature from the UMT MLM contrast runs (CPU, local).

Reads the token-level sparse arrays of umt_mlm_feature_contrast (R / C / Rev) and writes, per
clip x condition: total activation z, MLM contribution (dT_ab/dz * z, + = toward word a),
activation per frame position and per patch. Input for the visualisation/umt_mlm_feature_*.py figures.
Usage: FEATURE=4250 uv run python src/stage3_analysis/umt_mlm_feature_profile.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = {
    "feature":  int(os.environ.get("FEATURE", "4250")),
    # every MLM run the figures may draw on; add new run dirs here and all scripts pick them up
    "in_dirs":  [ROOT / "outputs/analysis/umt_mlm_feature_contrast",
                 ROOT / "outputs/analysis/umt_mlm_feature_contrast_pushpull"],
    "ablation_dirs": [ROOT / "outputs/analysis/umt_mlm_group_ablation",
                      ROOT / "outputs/analysis/umt_mlm_group_ablation_pushpull"],
    "n_frames": 12, "n_patches": 196,
}


def profile_dir(feature):
    return ROOT / "outputs/analysis/umt_mlm_feature_profile" / f"f{feature}"


def clip_profile(path, cfg):
    """One sparse npz -> (z total, contribution total, z per frame position, z per patch)."""
    with np.load(path) as s:
        m = s["lat"] == cfg["feature"]
        z, c = s["z"][m].astype(np.float32), s["dfa"][m]
        pos, patch = s["pos"][m].astype(int), s["patch"][m].astype(int)
    return (float(z.sum()), float(c.sum()), np.bincount(pos, z, cfg["n_frames"]),
            np.bincount(patch, z, cfg["n_patches"]))


def run_dir(in_dir, cfg):
    """All clip x condition rows of one contrast run (rows without a sparse file are skipped)."""
    rows = []
    for r in pd.read_parquet(in_dir / "per_clip.parquet").itertuples():
        path = in_dir / "sparse" / r.condition / f"{r.clip_id}.npz"
        if path.exists():
            z, c, z_pos, z_patch = clip_profile(path, cfg)
            rows.append(dict(clip_id=int(r.clip_id), a=r.a, b=r.b, true_class=r.true_class,
                             condition=r.condition, split=r.split, T_ab=r.T_ab, z=z, contrib=c,
                             z_pos=z_pos, z_patch=z_patch, source=in_dir.name))
    return rows


def find_run(pair, dirs, table):
    """The first run dir whose `table` parquet holds pair (a, b); error if none does."""
    for d in dirs:
        t = pd.read_parquet(d / table, columns=["a", "b"])
        if ((t.a == pair[0]) & (t.b == pair[1])).any():
            return d
    raise ValueError(f"pair {pair} not in any of {[d.name for d in dirs]} ({table})")


def build_profile(feature, cfg=CFG):
    """Write (and return) clip_profile.parquet for one feature over all MLM contrast runs."""
    cfg = dict(cfg, feature=feature)
    df = pd.DataFrame([r for d in cfg["in_dirs"] for r in run_dir(d, cfg)])
    df = df.drop_duplicates(["clip_id", "a", "b", "condition"])
    profile_dir(feature).mkdir(parents=True, exist_ok=True)
    df.to_parquet(profile_dir(feature) / "clip_profile.parquet")
    print(f"feature {feature}: {len(df)} clip-conditions -> {profile_dir(feature)}")
    return df


def load_profile(feature):
    """clip_profile.parquet for a feature, building it first if it doesn't exist yet."""
    path = profile_dir(feature) / "clip_profile.parquet"
    return pd.read_parquet(path) if path.exists() else build_profile(feature)


def main():
    build_profile(CFG["feature"])


if __name__ == "__main__":
    main()
