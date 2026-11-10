"""
Scan all L5 SAE features for an odd/even frame "comb": correlation of the class-averaged frame shape
with (-1)^frame, plus position lock under Rev and firing breadth. Writes comb_scan_L5.csv.
Usage: SAE_LAYER=5 uv run python src/stage3_analysis/umt_comb_scan.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "prof_dir": ROOT / f"outputs/analysis/umt_layer_positional_ablation_L{L}_splice",
    "sets_csv": ROOT / f"outputs/analysis/umt_layer_positional/sets_L{L}.csv",
    "out_csv": ROOT / f"outputs/analysis/umt_layer_positional/comb_scan_L{L}.csv",
    "early": 8, "cuts": (0.5, 0.7, 0.8), "min_fire": 0.5, "show": 25,
}


def class_shapes(cfg, cond):
    """(n_classes, 12, F) class-mean profiles normalised to their own mean; plus clip firing rate (F,)."""
    with np.load(str(cfg["prof_dir"] / f"profiles_{cond}.npz")) as z:
        cls, p = z["class_id"], z["prof"].astype(np.float32)
    P = np.stack([p[cls == c].mean(0) for c in np.unique(cls)])
    fire = (p.sum(1) > 0).mean(0)
    return P / np.maximum(P.mean(1, keepdims=True), 1e-12), fire


def rows_corr(X, v):
    """Pearson r of each column of X (n, F) with vector v (n,)."""
    x, v = X - X.mean(0), v - v.mean()
    return (x * v[:, None]).sum(0) / np.sqrt((x ** 2).sum(0) * (v ** 2).sum() + 1e-12)


def lock_rev(R, Rev):
    """Median over classes of corr(R shape, Rev shape as is), per feature."""
    a, b = R - R.mean(1, keepdims=True), Rev - Rev.mean(1, keepdims=True)
    return np.median((a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12), 0)


def main(cfg):
    R, fire = class_shapes(cfg, "R")
    Rev, _ = class_shapes(cfg, "Rev")
    e, par = cfg["early"], np.array([-1.0, 1.0] * 6)              # +1 on odd frames
    g = R.mean(0)                                                   # (12, F)
    cls_r = np.stack([rows_corr(R[c, :e], par[:e]) for c in range(len(R))])
    t = pd.DataFrame({"feature": np.arange(g.shape[1]), "fire": fire, "comb_early": rows_corr(g[:e], par[:e]),
                      "comb_all": rows_corr(g, par), "comb_cls": np.median(cls_r, 0), "lock_rev": lock_rev(R, Rev)})
    t = t.merge(pd.read_csv(cfg["sets_csv"])[["feature", "arm", "vtc_rank"]], on="feature", how="left")
    t["shape"] = [" ".join(f"{v:.1f}" for v in g[:, i]) for i in range(g.shape[1])]
    t.round(3).to_csv(cfg["out_csv"], index=False)
    f = t[t.fire >= cfg["min_fire"]]
    print(f"features firing on >= {cfg['min_fire']:.0%} of clips: {len(f)} / {len(t)}")
    for c in cfg["cuts"]:
        print(f"|comb_early| >= {c}: odd-high {(f.comb_early >= c).sum()}, even-high {(f.comb_early <= -c).sum()}")
    top = f.reindex(f.comb_early.abs().sort_values(ascending=False).index).head(cfg["show"])
    print(top.drop(columns="fire").round(2).to_string(index=False, na_rep="-"))


if __name__ == "__main__":
    main(CFG)
