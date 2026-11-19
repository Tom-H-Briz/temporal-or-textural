"""Shuffle + reversal sensitive vs insensitive latents: how much pure PE / PE x content / content DFA mass each holds
(19/11/26). Sensitivity per latent = median over clips of ||p_X - p_R|| / (||p_R|| + ||p_X||) on the 12-frame profile
(0 = unchanged, 1 = disjoint; flat profiles that stay flat count as insensitive). X = C (shuffle) or Rev.
Groups (active latents, per layer): sensitive = top third on both; insensitive = bottom third on both.
Inputs: umt_layer_pe_characterise L{L}.npz + features_L{L}.csv (from umt_layer_pe_characterise_analyse.py). CPU, local.
Usage: uv run python src/stage3_analysis/umt_order_sensitivity_mass.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from umt_layer_specificity import CFG as SPEC

D = Path(__file__).resolve().parents[2] / "outputs/analysis/umt_layer_pe_characterise"
CFG = {"dir": D, "layers": (5, 7, 9), "active": 0.5, "third": (1 / 3, 2 / 3), "n_bins": 10, "groups": Path(__file__).with_name("umt_order_groups.json"), "out": D / "order_sensitivity_mass.csv"}


def sensitivity(R, X):
    """Per latent: median over clips (where either profile fires) of the normalised profile distance."""
    num = np.linalg.norm(X - R, axis=1)
    den = np.linalg.norm(R, axis=1) + np.linalg.norm(X, axis=1)
    return np.nanmedian(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), 0)


def signed_mass(cfg):
    """{layer: (pos, neg) per latent}: class-mean signed DFA (all-174 thrfix), summed over classes where it is + / -."""
    d = {L: pd.read_parquet(str(SPEC["dfa"]).format(L), columns=["clip_id", "class_id", "signed_vec_R"])
         for L in cfg["layers"]}
    common = sorted(set.intersection(*[set(x.clip_id) for x in d.values()]))   # same clips as umt_layer_specificity
    out = {}
    for L, x in d.items():
        x = x.set_index("clip_id").loc[common]
        M = pd.DataFrame(np.stack(x.signed_vec_R.to_numpy())).groupby(x.class_id.to_numpy()).mean().to_numpy()
        out[L] = (np.clip(M, 0, None).sum(0), -np.clip(M, None, 0).sum(0))
    return out


def layer_table(L, cfg, sm):
    """Per group: latents, DFA mass, and the share of that mass in each PE class."""
    z = np.load(cfg["dir"] / f"L{L}.npz")
    R, C, Rev = (z[k].astype(np.float32) for k in ("R", "C", "Rev"))
    d = pd.read_csv(cfg["dir"] / f"features_L{L}.csv").assign(s_shuf=sensitivity(R, C), s_rev=sensitivity(R, Rev))
    a = d[(d.cls != "dead") & (d.fire_rate >= cfg["active"])].copy()
    a["dens"] = pd.qcut((R > 0).mean((0, 1))[a.latent], cfg["n_bins"], labels=False)   # per-frame firing rate
    for s in ("s_shuf", "s_rev"):                    # raw score tracks sparsity (rho -0.97): rank within density bin
        a[s] = a.groupby("dens")[s].rank(pct=True)
    lo, hi = cfg["third"]
    a["grp"] = np.where((a.s_shuf >= a.s_shuf.quantile(hi)) & (a.s_rev >= a.s_rev.quantile(hi)), "sensitive",
                np.where((a.s_shuf <= a.s_shuf.quantile(lo)) & (a.s_rev <= a.s_rev.quantile(lo)), "insensitive", "other"))
    a["pos"], a["neg"] = (v[a.latent.to_numpy()] for v in sm[L])
    rows = []
    for (g, b), sign in ((gb, s) for gb in a.groupby("grp") for s in ("dfa_mass", "pos", "neg")):
        m = b.groupby("cls")[sign].sum()
        rows.append({"layer": L, "grp": g, "mass": {"dfa_mass": "abs"}.get(sign, sign), "n": len(b),
                     "pct_all_mass": 100 * b[sign].sum() / a[sign].sum(),
                     **{f"pct_{c}": 100 * m.get(c, 0) / m.sum() for c in ("pe", "mixed", "content")}})
    sets = {g: sorted(map(int, b.latent)) for g, b in a.groupby("grp")}
    cells = a[a.grp != "other"].groupby(["grp", "cls"])                  # group x PE pool, for the patched arms
    return pd.DataFrame(rows), {**sets, **{f"{g}|{c}": sorted(map(int, b.latent)) for (g, c), b in cells}}


if __name__ == "__main__":
    sm = signed_mass(CFG)
    res = {L: layer_table(L, CFG, sm) for L in CFG["layers"]}
    t = pd.concat([r[0] for r in res.values()])
    CFG["groups"].write_text(json.dumps({str(L): r[1] for L, r in res.items()}))   # read by the frozen-PE GPU job
    t.round(3).to_csv(CFG["out"], index=False); print(t.round(2).to_string(index=False))
