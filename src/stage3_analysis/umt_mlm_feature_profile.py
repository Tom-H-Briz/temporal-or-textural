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
    "in_dirs":  [ROOT / "outputs/analysis/umt_mlm_feature_contrast",
                 ROOT / "outputs/analysis/umt_mlm_feature_contrast_pushpull"],
    "n_frames": 12, "n_patches": 196,
}
CFG["out_dir"] = ROOT / "outputs/analysis/umt_mlm_feature_profile" / f"f{CFG['feature']}"


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


def main():
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r for d in cfg["in_dirs"] for r in run_dir(d, cfg)])
    df = df.drop_duplicates(["clip_id", "a", "b", "condition"])
    out = cfg["out_dir"] / "clip_profile.parquet"
    df.to_parquet(out)
    print(f"feature {cfg['feature']}: {len(df)} clip-conditions -> {out}")


if __name__ == "__main__":
    main()
