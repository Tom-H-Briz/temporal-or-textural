"""
Are the QK-resonance clock candidates content-specific timing features? Per feature, from the post-fix
L5 per-clip frame profiles (35 SL classes): firing breadth, class concentration, per-class ramp_r under
R and Rev. Usage: uv run python src/stage3_analysis/umt_qk_candidate_profiles.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "features": [int(f) for f in os.environ.get("FEATURES", "3982,3999,3718,5422,3453").split(",")],
    "prof_dir": ROOT / f"outputs/analysis/umt_layer_positional_ablation_L{L}_splice",
    "labels": ROOT / "data/ssv2/labels/labels.json",
    "min_fire": 0.5, "top_classes": 6,               # class "fires" if >= half its clips are active
}


def load(cfg, cond):
    """(class_id (n,), prof (n, 12, 6144) float32) for one condition."""
    with np.load(str(cfg["prof_dir"] / f"profiles_{cond}.npz")) as z:
        return z["class_id"], z["prof"].astype(np.float32)


def ramp_r(p):
    """Pearson r of a 12-frame profile with frame index (nan if flat)."""
    return np.corrcoef(p, np.arange(len(p)))[0, 1] if p.std() > 0 else np.nan


def class_stats(f, cls, R, Rev):
    """Per class: fire rate, share of the feature's mass, ramp_r of class-mean profile (R, Rev)."""
    rows = []
    for c in np.unique(cls):
        m = cls == c
        rows.append({"class_id": c, "fire": (R[m, :, f].sum(1) > 0).mean(), "mass": R[m, :, f].sum(),
                     "ramp_R": ramp_r(R[m, :, f].mean(0)), "ramp_Rev": ramp_r(Rev[m, :, f].mean(0))})
    t = pd.DataFrame(rows)
    return t.assign(share=t.mass / t.mass.sum()).sort_values("mass", ascending=False)


def summary(f, cls, R, t, cfg):
    """One row: breadth, concentration, mass-weighted ramp_r under R and Rev."""
    w = t.share.fillna(0)
    return {"feature": f, "clips_firing": round((R[:, :, f].sum(1) > 0).mean(), 2),
            "classes_firing": int((t.fire >= cfg["min_fire"]).sum()), "of": len(t),
            "top3_share": round(t.share.head(3).sum(), 2),
            "ramp_R": round((w * t.ramp_R.fillna(0)).sum(), 2), "ramp_Rev": round((w * t.ramp_Rev.fillna(0)).sum(), 2)}


def main(cfg):
    names = {int(v): k for k, v in json.load(open(cfg["labels"])).items()}
    cls, R = load(cfg, "R")
    cls_rev, Rev = load(cfg, "Rev")
    assert (cls == cls_rev).all(), "R / Rev clip order differs"
    rows = []
    for f in cfg["features"]:
        t = class_stats(f, cls, R, Rev)
        rows.append(summary(f, cls, R, t, cfg))
        top = t.head(cfg["top_classes"]).assign(name=lambda d: d.class_id.map(names).str[:45])
        print(f"\n#{f} top classes by mass\n{top.round(2).to_string(index=False)}")
    print(f"\n{pd.DataFrame(rows).to_string(index=False)}")


if __name__ == "__main__":
    main(CFG)
