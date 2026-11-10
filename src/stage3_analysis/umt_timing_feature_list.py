"""
Known timing features at one layer (ramps, near-miss ramps, positional nonramps, manual position additions)
with post-fix shape stats: peak frame, ramp_r, position lock under Rev. Writes timing_features_L{L}.csv.
Usage: SAE_LAYER=5 uv run python src/stage3_analysis/umt_timing_feature_list.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "arms": {"ramp": "ramp", "near_miss_ramp": "ramp (near miss)", "nonramp": "positional nonramp"},
    "manual": {531: ("position", "mid-clip hump, f11≈0; selection said content_timed (rev_margin "
                                 "misfires on a symmetric hump), QK 10/11")} if L == 5 else {},
    "sets_csv": ROOT / f"outputs/analysis/umt_layer_positional/sets_L{L}.csv",
    "prof_dir": ROOT / f"outputs/analysis/umt_layer_positional_ablation_L{L}_splice",
    "out_csv": ROOT / f"outputs/analysis/umt_layer_positional/timing_features_L{L}.csv",
}


def members(cfg):
    """feature, group, note — from the selection arms plus manual additions."""
    s = pd.read_csv(cfg["sets_csv"])
    t = s[s.arm.isin(cfg["arms"])][["feature", "arm", "vtc_rank"]]
    t = t.assign(group=t.arm.map(cfg["arms"]), note="")
    if not cfg["manual"]:
        return t.drop(columns="arm").reset_index(drop=True)
    extra = pd.DataFrame([{"feature": f, "group": g, "note": n} for f, (g, n) in cfg["manual"].items()])
    extra = extra.merge(s[["feature", "vtc_rank"]], on="feature", how="left")
    return pd.concat([t.drop(columns="arm"), extra], ignore_index=True)


def class_shapes(cfg, cond, feats):
    """(n_classes, 12, k) class-mean profiles, each normalised to its own mean."""
    with np.load(str(cfg["prof_dir"] / f"profiles_{cond}.npz")) as z:
        cls, p = z["class_id"], z["prof"][:, :, feats].astype(np.float32)
    P = np.stack([p[cls == c].mean(0) for c in np.unique(cls)])
    return P / np.maximum(P.mean(1, keepdims=True), 1e-12)


def med_corr(A, B):
    """Median over classes of the per-class profile correlation, per feature: (n_cls, 12, k) -> (k,)."""
    a, b = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    r = (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)
    return np.median(r, 0)


def main(cfg):
    t = members(cfg)
    feats = t.feature.astype(int).tolist()
    R, Rev = class_shapes(cfg, "R", feats), class_shapes(cfg, "Rev", feats)
    g = R.mean(0)                                                  # (12, k) grand shape
    t["peak_frame"] = g.argmax(0)
    t["ramp_r"] = [np.corrcoef(g[:, i], np.arange(12))[0, 1] for i in range(len(feats))]
    t["lock_rev"], t["follow_rev"] = med_corr(R, Rev), med_corr(R, Rev[:, ::-1])
    t["shape"] = [" ".join(f"{v:.1f}" for v in g[:, i]) for i in range(len(feats))]
    t = t.round(2).sort_values(["group", "peak_frame"])
    t.to_csv(cfg["out_csv"], index=False)
    print(t.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main(CFG)
