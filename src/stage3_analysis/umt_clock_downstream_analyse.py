"""
Local analysis of umt_clock_downstream_ablation.py outputs (CPU). Priorities (user, 10 Nov):
  1. what consumes the timing features: per downstream latent, mass drop under clock (vs batch_base); are the pair
     key features (umt_pair_contrast_L{c}/features.csv) enriched among the most consumed; home-clip specificity;
     shape (ramp_r) vs size for key features and the L9 gated ramps; pair accuracy per arm.
  2. combs, real or artifact: where the high-norm tokens sit, same place every clip?, R vs Rev frame index,
     comb-map mass on them, persistence across layers, attention received by comb frames.
  Plus attention "stay near own time": diagonal / near-diagonal mass, batch_base vs clock.
Every table: all clips and non-selection clips (role != sel) side by side. Writes CSVs to <run dir>/analysis/.
Usage: LAYER=5 [RUN_DIR=...] uv run python src/stage3_analysis/umt_clock_downstream_analyse.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("LAYER", 5))
A = ROOT / "outputs/analysis"
CFG = {
    "layer": L, "capture": [c for c in (7, 9) if c > L],
    "run_dir": Path(os.environ.get("RUN_DIR", str(A / f"umt_clock_downstream_L{L}"))),
    "pairs": [(45, 43), (168, 165), (93, 86), (44, 41), (0, 32)],
    "pairs_csv": str(A / "umt_pair_contrast_L{c}/features.csv"),
    "timing_csv": str(A / "umt_layer_positional/timing_features_L{c}.csv"),
    "extra_sets": ROOT / "src/stage3_analysis/umt_clock_downstream_sets.json",
    "targets": {9: {393: 45, 4250: 43}},                         # L9 gated up / down ramps -> home class
    "trace": {7: [2983, 6110, 5621, 5806, 6113], 9: [1540, 5825, 3537, 65, 4903]},
    "arms_cmp": ["clock", "ramp", "positional", "all"], "min_mass_pct": 75, "top_n": 25,
    "n_perm": 10_000, "seed": 0, "norm_mult": 5.0,               # high-norm token: > 5 x the clip's median
}


def load(cfg, name, cond):
    """Memmap one stored array (read-only)."""
    return np.load(cfg["run_dir"] / f"{name}_{cond}.npy", mmap_mode="r")


def meta(cfg):
    """clips (with done flags for R), arm name -> row index."""
    clips = pd.read_csv(cfg["run_dir"] / "clips.csv", dtype={"clip_id": str})
    clips["done"] = np.asarray(load(cfg, "done", "R")) & np.asarray(load(cfg, "done", "Rev"))
    arms = pd.read_csv(cfg["run_dir"] / "arms.csv")
    return clips, dict(zip(arms.arm, arms.row))


def comb_flags(base):
    """(F,) bool: comb rule of umt_comb_scan.py on baseline profiles (n, 12, F)."""
    fire = (base.sum(1) > 0).mean(0)
    g = base.mean(0)
    g = g / np.maximum(g.mean(0, keepdims=True), 1e-12)
    par = np.array([-1.0, 1.0] * 4)
    x = g[:8] - g[:8].mean(0)
    r = (x * (par - par.mean())[:, None]).sum(0) / np.sqrt((x ** 2).sum(0) * ((par - par.mean()) ** 2).sum() + 1e-12)
    oe = g[[1, 3, 5, 7]].mean(0) / np.maximum(g[[0, 2, 4, 6]].mean(0), 1e-12)
    return (fire >= 0.5) & (r >= 0.7) & (oe >= 3)


def labels(cfg, c, base):
    """Per capture-layer latent: key (home classes of our pairs), timing group, trace / target / comb flags."""
    ours = {x for p in cfg["pairs"] for x in p}
    k = pd.read_csv(cfg["pairs_csv"].format(c=c))
    k = k[k.home.isin(ours)].groupby("latent").home.apply(lambda s: ",".join(map(str, sorted(set(s)))))
    t = pd.read_csv(cfg["timing_csv"].format(c=c)).set_index("feature").group
    lab = pd.DataFrame({"feature": np.arange(base.shape[-1])})
    lab["key"], lab["timing"] = lab.feature.map(k).fillna(""), lab.feature.map(t).fillna("")
    lab["trace"] = lab.feature.isin(cfg["trace"].get(c, []))
    lab["target"] = lab.feature.isin(list(cfg["targets"].get(c, {})))
    lab["comb"] = comb_flags(base)
    return lab


def arm_prof(cfg, c, cond, row, mask):
    """(n_mask, 12, F) float32 profiles of one arm row at capture layer c."""
    return np.asarray(load(cfg, f"prof_L{c}", cond)[:, row], dtype=np.float32)[mask]


def consumption(cfg, c, cond, rows, mask):
    """Per latent: batch_base mass, and per compared arm the relative and absolute mass drop."""
    base = arm_prof(cfg, c, cond, rows["batch_base"], mask).sum((0, 1))
    t = pd.DataFrame({"feature": np.arange(len(base)), "base_mass": base})
    for a in cfg["arms_cmp"]:
        m = arm_prof(cfg, c, cond, rows[a], mask).sum((0, 1))
        t[f"drop_{a}"] = 1 - m / np.maximum(base, 1e-12)
        t[f"absdrop_{a}"] = base - m
    nz = base[base > 0]
    t["active"] = base > (np.percentile(nz, cfg["min_mass_pct"]) if len(nz) else 0)
    return t


def enrichment(cfg, t, lab, arm="clock"):
    """Median consumption percentile (among active latents) per label group, with a permutation p."""
    a = t[t.active].merge(lab, on="feature")
    a["pct"] = a[f"drop_{arm}"].rank(pct=True) * 100
    rng, rows = np.random.default_rng(cfg["seed"]), []
    groups = {"key": a.key != "", "trace": a.trace, "timing": a.timing != "", "comb": a.comb, "target": a.target}
    for g, m in groups.items():
        n = int(m.sum())
        if n == 0:
            continue
        obs = a.pct[m].median()
        null = np.array([np.median(rng.choice(a.pct.values, n, replace=False)) for _ in range(cfg["n_perm"])])
        rows.append({"group": g, "n_active": n, "median_pct": round(obs, 1), "p_greater": (null >= obs).mean()})
    return pd.DataFrame(rows), a


def key_table(a, arm="clock"):
    """Every pair key feature, trace candidate and target among active latents, most consumed first."""
    k = a[(a.key != "") | a.trace | a.target]
    cols = ["feature", "key", "timing", "trace", "target", "comb", "base_mass", f"drop_{arm}", "pct"]
    return k.sort_values(f"drop_{arm}", ascending=False)[cols + [c for c in a if c.startswith("drop_") and c not in cols]]


def ramp_r(p):
    return float(np.corrcoef(p, np.arange(len(p)))[0, 1]) if p.std() > 0 else np.nan


def home_shape(cfg, c, cond, rows, clips, feats_home, mask, arm="clock"):
    """Per (feature, home): clock drop on home vs other pair clips; home-clip ramp_r and size, base vs arm."""
    B = arm_prof(cfg, c, cond, rows["batch_base"], mask)
    X = arm_prof(cfg, c, cond, rows[arm], mask)
    cls, out = clips.class_id.values[mask], []
    for f, homes in feats_home:
        h = np.isin(cls, homes)
        if h.sum() == 0 or (~h).sum() == 0:
            continue
        drop = lambda m: 1 - X[m, :, f].sum() / B[m, :, f].sum() if B[m, :, f].sum() > 1e-3 else np.nan
        pb, px = B[h, :, f].mean(0), X[h, :, f].mean(0)
        out.append({"feature": f, "home": ",".join(map(str, homes)), "n_home": int(h.sum()),
                    "drop_home": drop(h), "drop_other": drop(~h), "ramp_r_base": ramp_r(pb), "ramp_r_arm": ramp_r(px),
                    "size_ratio": px.sum() / max(pb.sum(), 1e-12),
                    "shape_base": " ".join(f"{v:.0f}" for v in pb), "shape_arm": " ".join(f"{v:.0f}" for v in px)})
    return pd.DataFrame(out)


def pair_accuracy(cfg, cond, clips, rows, mask):
    """Per arm x pair: pair accuracy (true logit > opposite) and 174-way top-1, and both minus batch_base."""
    lg = np.asarray(load(cfg, "logits", cond), dtype=np.float32)[mask]
    cls = clips.class_id.values[mask]
    opp = {a: b for p in cfg["pairs"] for a, b in (p, p[::-1])}
    o = np.array([opp[c] for c in cls])
    out = []
    for arm, j in rows.items():
        x = lg[:, j]
        pa = x[np.arange(len(x)), cls] > x[np.arange(len(x)), o]
        top = x.argmax(1) == cls
        for p in cfg["pairs"]:
            m = np.isin(cls, p)
            mean = lambda v: v[m].mean() if m.any() else np.nan
            out.append({"arm": arm, "pair": f"{p[0]}/{p[1]}", "n": int(m.sum()), "pair_acc": mean(pa), "top1": mean(top)})
    t = pd.DataFrame(out)
    ref = t[t.arm == "batch_base"].set_index("pair")
    t["d_pair_acc"] = t.pair_acc - t.pair.map(ref.pair_acc)
    t["d_top1"] = t.top1 - t.pair.map(ref.top1)
    return t


def attention(cfg, cond, rows, mask, arms=("batch_base", "clock", "all")):
    """Per head: diagonal / near (|d|<=1) mass and mean |d frame| per arm; clock - batch_base paired per clip."""
    at = np.asarray(load(cfg, "attn", cond), dtype=np.float32)[mask]          # (n, R, H, 12, 12)
    d = np.abs(np.subtract.outer(np.arange(12), np.arange(12)))
    stat = lambda x: {"diag": (x * (d == 0)).sum((-1, -2)) / 12, "near": (x * (d <= 1)).sum((-1, -2)) / 12,
                      "dist": (x * d).sum((-1, -2)) / 12}                      # (n, H) each
    S = {a: stat(at[:, rows[a]]) for a in arms}
    out = []
    for h in range(at.shape[2]):
        r = {"head": h}
        for a in arms:
            r.update({f"{k}_{a}": S[a][k][:, h].mean() for k in ("diag", "near", "dist")})
        dd = S["clock"]["diag"][:, h] - S["batch_base"]["diag"][:, h]
        r.update(d_diag_clock=dd.mean(), frac_diag_down=(dd < 0).mean())
        out.append(r)
    received = at[:, rows["baseline"]].mean((0, 1, 2))                 # (12,) mass received per key frame
    return pd.DataFrame(out), received


def high_tokens(cfg, layer, cond, mask):
    """(n, 12, 196) bool: token norm > norm_mult x its clip's median, at one layer (baseline)."""
    n = np.asarray(load(cfg, f"norm_L{layer}", cond), dtype=np.float32)[mask]
    return (n > cfg["norm_mult"] * np.median(n, 1, keepdims=True)).reshape(-1, 12, 196), n.reshape(-1, 12, 196)


def combs(cfg, cond, mask):
    """High-norm tokens at LAYER: count, frame share, most stable positions, comb-map overlap, cross-layer persistence."""
    hi, nm = high_tokens(cfg, cfg["layer"], cond, mask)
    cm = np.asarray(load(cfg, "comb", cond), dtype=np.float32)[mask].reshape(-1, 12, 196)
    stab = hi.mean(0)                                            # (12, 196): share of clips high at each slot
    top = np.argsort(stab.ravel())[::-1][:15]
    tot = np.maximum(cm.sum((1, 2)), 1e-12)
    corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(nm, cm) if b.std() > 0]
    jac = {}
    for c in cfg["capture"]:
        h2, _ = high_tokens(cfg, c, cond, mask)
        inter, uni = (hi & h2).sum((1, 2)), np.maximum((hi | h2).sum((1, 2)), 1)
        jac[f"jaccard_L{cfg['layer']}_L{c}"] = float(np.median(inter / uni))
    return {"n_high_median": float(np.median(hi.sum((1, 2)))), "frame_share": hi.sum((0, 2)) / max(hi.sum(), 1),
            "stable_slots": [(int(i // 196), int(i % 196), round(float(stab.ravel()[i]), 3)) for i in top],
            "patch_map": hi.any(1).mean(0).reshape(14, 14), "comb_on_high": float(np.median((cm * hi).sum((1, 2)) / tot)),
            "corr_norm_comb": float(np.median(corr)) if corr else np.nan, **jac}


def report_consumption(cfg, clips, rows, out, cond, sub, mask):
    """Priority 1 per capture layer: enrichment, most consumed, key features, home shape vs size."""
    for c in cfg["capture"]:
        lab = labels(cfg, c, arm_prof(cfg, c, cond, rows["baseline"], mask))
        t = consumption(cfg, c, cond, rows, mask)
        enr, a = enrichment(cfg, t, lab)
        a.to_csv(out / f"consumption_L{c}_{cond}_{sub}.csv", index=False)
        print(f"\n######## L{cfg['layer']} clock -> L{c}  {cond}  {sub} ({int(mask.sum())} clips)\n{enr.to_string(index=False)}")
        cols = ["feature", "key", "timing", "trace", "comb", "base_mass", "drop_clock", "absdrop_clock", "drop_ramp", "drop_all"]
        for by, name in (("drop_clock", "relative"), ("absdrop_clock", "absolute mass")):
            top = a.sort_values(by, ascending=False).head(cfg["top_n"])
            print(f"most consumed by {name} (active):\n" + top[cols].round(3).to_string(index=False))
        print("pair key features / trace / targets:\n" + key_table(a).round(3).head(40).to_string(index=False))
        fh = [(f, [int(h) for h in k.split(",")]) for f, k in zip(lab.feature, lab.key) if k]
        fh += [(f, [h]) for f, h in cfg["targets"].get(c, {}).items()]
        hs = home_shape(cfg, c, cond, rows, clips, fh, mask)
        hs.to_csv(out / f"home_shape_L{c}_{cond}_{sub}.csv", index=False)
        if len(hs):
            print("home-clip shape vs size (clock):\n" + hs.sort_values("drop_home", ascending=False).round(2).to_string(index=False))


def report_rest(cfg, rows, clips, out, cond, sub, mask):
    """Pair accuracy, attention near own time, combs (+ attention received per key frame)."""
    pa = pair_accuracy(cfg, cond, clips, rows, mask)
    pa.to_csv(out / f"pair_acc_{cond}_{sub}.csv", index=False)
    print(f"\npair accuracy {cond} {sub} (minus batch_base):\n"
          + pa[pa.arm.isin(["batch_base", "clock", "ramp", "all"])].round(3).to_string(index=False))
    at, rec = attention(cfg, cond, rows, mask)
    at.to_csv(out / f"attention_block{cfg['layer'] + 1}_{cond}_{sub}.csv", index=False)
    print(f"\nblock {cfg['layer'] + 1} attention {cond} {sub}:\n" + at.round(4).to_string(index=False))
    cb = combs(cfg, cond, mask)
    print(f"\ncombs L{cfg['layer']} {cond} {sub}: " + json.dumps({k: v for k, v in cb.items() if k not in
          ("frame_share", "patch_map", "stable_slots")}))
    print("  high-norm frame share:", np.round(cb["frame_share"], 3), "\n  attention received per key frame:", np.round(rec, 3))
    print("  most stable (frame, patch, share of clips):", cb["stable_slots"][:10])
    print("  patch map (share of clips with a high token there, any frame):\n" + np.array2string(cb["patch_map"], precision=2))


def main(cfg):
    clips, rows = meta(cfg)
    out = cfg["run_dir"] / "analysis"
    out.mkdir(exist_ok=True)
    print(f"{int(clips.done.sum())}/{len(clips)} clips done under R and Rev; arms {list(rows)}")
    for cond in ("R", "Rev"):
        for sub, m in (("all", clips.done.values), ("clean", clips.done.values & (clips.role != "sel").values)):
            report_consumption(cfg, clips, rows, out, cond, sub, m)
            report_rest(cfg, rows, clips, out, cond, sub, m)


if __name__ == "__main__":
    main(CFG)
