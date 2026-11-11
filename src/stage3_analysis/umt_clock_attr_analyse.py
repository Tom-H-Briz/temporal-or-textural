"""
What builds the pair key features? Reads the raw clock-downstream run's attribution (attr_{R,Rev}.npy: per clip, its
home targets x frame x LAYER latent, activation x gradient, last column = error + dim_mean) and ranks contributors per
target, flagged clock / comb / L5 pair feature, with their QK resonance with the clock (seek / sought best |z|, as
umt_qk_content_timing.py), lead (source frame - target frame) and the R -> Rev sign. Linearity check: summed clock
attribution vs the clock arm's actual drop. Usage: LAYER=5|7 uv run python src/stage3_analysis/umt_clock_attr_analyse.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

L = int(os.environ.get("LAYER", 5))
os.environ.setdefault("SAE_LAYER", str(L))                      # umt_qk_resonance reads it at import
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "stage3_analysis"))
A = ROOT / "outputs/analysis"
CFG = {
    "run_dir": Path(os.environ.get("RUN_DIR", str(A / f"umt_clock_downstream_L{L}_raw"))),
    "sets": ROOT / "src/stage3_analysis/umt_clock_downstream_sets.json",
    "pair_feats": A / f"umt_pair_contrast_L{L}/features.csv",
    "top_n": 15, "F": 6144,
}


def load(cfg, name, cond):
    return np.load(cfg["run_dir"] / f"{name}_{cond}.npy", mmap_mode="r")


def target_attr(cfg, cond):
    """{(c, t): (mean attribution (12, F+1) over the target's clips, clip indices)}; a target may sit in a
    different slot for different classes, so sums run over every (slot, clips) where it appears."""
    at, tg = load(cfg, "attr", cond), load(cfg, "attr_tg", cond)
    acc = {}
    for k in range(tg.shape[1]):
        for key in {tuple(map(int, r)) for r in tg[:, k]}:
            idx = np.where((tg[:, k] == key).all(1))[0]
            s, ids = acc.get(key, (0.0, []))
            acc[key] = (s + at[idx, k].astype(np.float32).sum(0), ids + list(idx))
    return {key: (s / len(ids), np.array(sorted(ids))) for key, (s, ids) in acc.items()}


def flags(cfg):
    """Clock / comb lists (sets json) and the L5|L7 pair-contrast content features with their home pair."""
    s = json.load(open(cfg["sets"]))[str(L)]
    p = pd.read_csv(cfg["pair_feats"]) if cfg["pair_feats"].exists() else pd.DataFrame(columns=["latent", "object"])
    return set(s["clock"]), set(s["comb"]), p.groupby("latent").object.agg(lambda x: ",".join(sorted(set(x))))


def resonance(clock):
    """Per LAYER latent: signed best seek / sought z vs the clock and the |z| percentile over the dictionary."""
    import umt_qk_content_timing as qct
    seek, sought = qct.timing_z(qct.CFG, sorted(clock))
    out = {}
    for name, z in (("seek", seek), ("sought", sought)):
        mag, signed, _, _ = qct.best(z)
        out[name], out[f"{name}_pct"] = signed, 100 * mag.argsort().argsort() / (len(mag) - 1)
    return pd.DataFrame(out)


def target_frames(cfg, cond, c, t, idx):
    """Baseline (row 0) frame profile of target t at layer c over its clips, mean -> (12,)."""
    return load(cfg, f"prof_L{c}", cond)[idx, 0, :, t].astype(np.float32).mean(0)


def contributors(cfg, key, M, Mrev, tf, fl, res):
    """Top-n LAYER latents by |clip-summed attribution| for one target; share of total |attribution|,
    lead = |attr|-weighted source frame - target-profile mean frame (frames), other-condition attribution + sign flip."""
    clock, comb, pair = fl
    tot, fr = M[:, :-1].sum(0), np.arange(12)
    tgt_frame = (tf * fr).sum() / max(tf.sum(), 1e-6)
    rows = []
    for f in np.argsort(-np.abs(tot))[:cfg["top_n"]]:
        w = np.abs(M[:, f])
        oth = float(Mrev[:, f].sum()) if Mrev is not None else np.nan
        rows.append({"layer": key[0], "target": key[1], "latent": int(f), "attr": float(tot[f]),
                     "share": float(tot[f] / np.abs(tot).sum()), "lead": float((w * fr).sum() / w.sum() - tgt_frame),
                     "attr_other": oth, "flip": bool(np.sign(oth) != np.sign(tot[f])), "clock": int(f) in clock,
                     "comb": int(f) in comb, "pair": pair.get(int(f), ""), **res.iloc[int(f)].to_dict()})
    return rows


def linearity(cfg, cond, key, M, idx, clock):
    """Predicted (summed clock attribution) vs actual (batch_base - clock arm) clip-summed drop of the target."""
    arms = pd.read_csv(cfg["run_dir"] / "arms.csv").set_index("arm").row
    p = load(cfg, f"prof_L{key[0]}", cond)
    act = (p[idx, arms["batch_base"], :, key[1]].astype(np.float32) - p[idx, arms["clock"], :, key[1]]).sum(1).mean()
    return {"layer": key[0], "target": key[1], "cond": cond, "pred_drop": float(M[:, sorted(clock)].sum()),
            "actual_drop": float(act), "remainder": float(M[:, -1].sum()), "dict_total": float(M[:, :-1].sum())}


def main(cfg):
    fl, out = flags(cfg), cfg["run_dir"] / "analysis"
    out.mkdir(exist_ok=True)
    res, ta = resonance(fl[0]), {c: target_attr(cfg, c) for c in ("R", "Rev")}
    for cond, other in (("R", "Rev"), ("Rev", "R")):
        rows, lin = [], []
        for key, (M, idx) in sorted(ta[cond].items()):
            tf = target_frames(cfg, cond, *key, idx)
            rows += contributors(cfg, key, M, ta[other].get(key, (None,))[0], tf, fl, res)
            lin.append(linearity(cfg, cond, key, M, idx, fl[0]))
        tab, lin = pd.DataFrame(rows), pd.DataFrame(lin)
        tab.to_csv(out / f"attr_contributors_{cond}.csv", index=False)
        lin.to_csv(out / f"attr_linearity_{cond}.csv", index=False)
        report(tab, lin, cond)


def report(tab, lin, cond):
    """Linearity per target; top-8 contributors per target; pooled QK / flip / lead stats for the non-clock top-n."""
    pd.set_option("display.width", 220)
    print(f"\n===== {cond}: clock attribution (pred) vs clock-arm drop (actual), clip-summed =====")
    print(lin.round(1).to_string(index=False))
    cols = ["latent", "attr", "share", "lead", "attr_other", "flip", "clock", "comb", "pair", "seek_pct", "sought_pct"]
    for (c, t), g in tab.groupby(["layer", "target"]):
        print(f"\n-- L{c} #{t} ({cond}) top 8"); print(g[cols].head(8).round(3).to_string(index=False))
    nc = tab[~tab.clock]
    print(f"\n{cond} pooled non-clock top contributors (n={len(nc)}): median seek_pct {nc.seek_pct.median():.0f}, "
          f"sought_pct {nc.sought_pct.median():.0f} (dictionary median 50); flip {100 * nc.flip.mean():.0f}%; "
          f"|lead| > 1 frame {100 * (nc.lead.abs() > 1).mean():.0f}%; comb {nc.comb.sum()}; pair {(nc.pair != '').sum()}")
    print("most frequent non-clock contributors:", nc.latent.value_counts().head(12).to_dict())


if __name__ == "__main__":
    main(CFG)
