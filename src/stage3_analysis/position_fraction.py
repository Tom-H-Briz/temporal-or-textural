"""
How much of each SAE decoder direction is position? (agreed 11/11/26). Data ruler from the clock-downstream run's
res_L{l}_{cond}.npy (baseline residual per frame / tubelet, mean over space): per class, position means minus their
mean over positions, averaged over classes -> P (T, D). C1 (shuffled) is the ruler: content is spread over every slot,
so what is left is the slot itself; R is compared against it. Per decoder direction (unit):
  narrow = best cos^2 with one position vector P_t (and which t)    broad = share in the (T-1)-d span of P
  smooth = share in the degree-1..2 time-trend part (ramp-like)     sharp = broad - smooth (spike-like)
Random 768-d direction: broad ~ (T-1)/768. Percentiles are over the whole dictionary.
Usage: [MODEL=videomae] [RUN_DIR=...] uv run python src/stage3_analysis/position_fraction.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
VM = os.environ.get("MODEL", "umt") == "videomae"
A = ROOT / "outputs/analysis"
CFG = {
    "run_dir": Path(os.environ.get("RUN_DIR", str(A / f"{'vm' if VM else 'umt'}_clock_downstream_L5_raw"))),
    "layers": [5, 7, 9], "smooth_deg": 2,
    "ruler_cond": "C1" if VM else "R", "compare_cond": "R",    # UMT run has no shuffled pass: R only (flagged)
    "sae": str(ROOT / f"outputs/sae/sae_{'vmae' if VM else 'umt'}_ssv2_k64_x8_l{{l}}_job7ep_best.pt"),
    "sets": ROOT / f"src/stage3_analysis/{'vm' if VM else 'umt'}_clock_downstream_sets.json",
    "pairs_csv": str(A / f"{'videomae' if VM else 'umt'}_pair_contrast_L{{l}}/features.csv"),
    "gate_csv": str(A / "scaffold_selection/L{l}_x8k64_VM.csv") if VM else "",
}


def load_res(cfg, l, cond):
    """(n_done, T, D) float32 residual per position and the clips' class ids."""
    clips = pd.read_csv(cfg["run_dir"] / "clips.csv")
    done = np.load(cfg["run_dir"] / f"done_{cond}.npy").astype(bool)
    r = np.load(cfg["run_dir"] / f"res_L{l}_{cond}.npy", mmap_mode="r")
    return np.asarray(r[done], dtype=np.float32), clips.class_id.values[done]


def ruler(res, cls):
    """P (T, D): per-class position means, centred over positions, averaged over classes (each class weighs
    the same). Also the two halves of the classes (alternate class ids) for a stability check."""
    per = {c: res[cls == c].mean(0) for c in np.unique(cls)}
    per = {c: p - p.mean(0, keepdims=True) for c, p in per.items()}
    ids = sorted(per)
    mean = lambda cs: np.mean([per[c] for c in cs], 0)
    if len(ids) < 2:                                             # one class (smoke runs): no split possible
        return mean(ids), mean(ids), mean(ids)
    return mean(ids), mean(ids[0::2]), mean(ids[1::2])


def pos_cos(P1, P2):
    """(T,) cosine between matching position vectors of two rulers."""
    return (P1 * P2).sum(1) / (np.linalg.norm(P1, axis=1) * np.linalg.norm(P2, axis=1) + 1e-12)


def time_basis(P, deg):
    """Orthonormal (D, T-1) basis of the span of P, ordered by time-polynomial degree: the first `deg` columns
    span the smooth (degree 1..deg trend) part, the rest the sharp remainder."""
    T = len(P)
    B, _ = np.linalg.qr(np.vander(np.arange(T) - (T - 1) / 2, T, increasing=True))   # col 0 = constant
    Q, _ = np.linalg.qr((B[:, 1:].T @ P).T)                     # (D, T-1), degree order kept
    return Q


def fractions(dec, P, Q, deg):
    """Per unit decoder direction: broad / smooth / sharp shares, best single-position cos^2 and its position,
    and each as a percentile over the dictionary."""
    d = dec / np.linalg.norm(dec, axis=1, keepdims=True)
    proj = (d @ Q) ** 2
    Pu = P / np.linalg.norm(P, axis=1, keepdims=True)
    c2 = (d @ Pu.T) ** 2                                         # (F, T)
    t = pd.DataFrame({"feature": np.arange(len(d)), "broad": proj.sum(1), "smooth": proj[:, :deg].sum(1),
                      "sharp": proj[:, deg:].sum(1), "narrow": c2.max(1), "narrow_pos": c2.argmax(1)})
    for k in ("broad", "smooth", "sharp", "narrow"):
        t[f"{k}_pct"] = t[k].rank(pct=True) * 100
    return t


def labels(cfg, l):
    """feature -> label: spikes / others (sets json, this layer), gate / near_miss (VM scaffold), pair:<homes>."""
    lab = {}
    p = Path(cfg["pairs_csv"].format(l=l))
    if p.exists():
        lab.update(pd.read_csv(p).groupby("latent").home.agg(lambda h: "pair:" + ",".join(map(str, sorted(set(h))))))
    g = cfg["gate_csv"].format(l=l)
    if g and Path(g).exists():
        s = pd.read_csv(g)
        lab.update({int(f): ("gate" if st == "member" else st) for f, st in zip(s.feature_idx, s.status)})
    for arm, fs in json.load(open(cfg["sets"])).get(str(l), {}).items():
        if arm != "comb":
            lab.update({int(f): arm for f in fs})
    return lab


def ruler_check(cfg, l):
    """Ruler P from ruler_cond; prints split-half stability and its agreement with compare_cond, per position."""
    P, h1, h2 = ruler(*load_res(cfg, l, cfg["ruler_cond"]))
    Pc = ruler(*load_res(cfg, l, cfg["compare_cond"]))[0]
    print(f"\n=== L{l} ruler {cfg['ruler_cond']} | |P_t| {np.round(np.linalg.norm(P, axis=1), 1)}")
    print(f"  split-half cos per position {np.round(pos_cos(h1, h2), 2)}")
    print(f"  {cfg['ruler_cond']} vs {cfg['compare_cond']} cos per position {np.round(pos_cos(P, Pc), 2)}")
    return P


def main(cfg):
    out = cfg["run_dir"] / "analysis"
    out.mkdir(exist_ok=True)
    if cfg["ruler_cond"] == cfg["compare_cond"]:
        print("NOTE: no shuffled pass in this run -> ruler = R (may include content typical of a time)")
    for l in cfg["layers"]:
        P = ruler_check(cfg, l)
        dec = torch.load(cfg["sae"].format(l=l), map_location="cpu", weights_only=True)["sae_state_dict"]
        t = fractions(dec["dictionary._weights"].float().numpy(), P, time_basis(P, cfg["smooth_deg"]), cfg["smooth_deg"])
        lab = labels(cfg, l)
        t["label"] = t.feature.map(lab).fillna("")
        t.round(4).to_csv(out / f"position_fraction_L{l}.csv", index=False)
        report(t, len(P), dec["dictionary._weights"].shape[1])


def report(t, T, D):
    """Median shares per label group (pair features pooled) and every labelled non-pair feature."""
    grp = t.label.where(~t.label.str.startswith("pair"), "pair").replace("", "unlabelled")
    print(f"  random-direction broad ~ {(T - 1) / D:.4f}; dictionary median broad {t.broad.median():.4f}")
    cols = ["broad", "smooth", "sharp", "narrow", "broad_pct", "narrow_pct"]
    print(t.groupby(grp)[cols].median().round(3).assign(n=grp.value_counts()).to_string())
    k = t[(t.label != "") & ~t.label.str.startswith("pair")].sort_values("broad", ascending=False)
    print(k[["feature", "label", "narrow_pos"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
