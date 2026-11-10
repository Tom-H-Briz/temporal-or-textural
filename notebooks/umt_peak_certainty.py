"""UMT noun / verb peak-layer certainty (09/11/26): per unit, per-layer effect with 95% CI and P(layer is peak),
bootstrapping test rows paired across layers (sets.csv is shared, so the same rows appear at every layer).
Raw edit - baseline (controls broken in this run). Read by noun_verb_scaleup_tables.py (Layers sheet).
Usage: uv run python notebooks/umt_peak_certainty.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

A = Path(__file__).resolve().parents[1] / "outputs" / "analysis"
OUT = A / "noun_verb_scaleup" / "peak_certainty.csv"
NB, LAYERS, rng = 2000, (5, 7, 9), np.random.default_rng(0)


def main_edit(edits, unit):
    n = sorted(e for e in set(edits) if e.startswith(unit + "_") and e[len(unit) + 1:].replace("group", "0").isdigit())
    return f"{unit}_group" if f"{unit}_group" in n else (n[0] if n else None)


def boot(cols, stat):
    """cols: {L: per-row arrays (aligned rows)}; stat(arrays at idx) -> value. Returns point, CI, P(peak) per layer."""
    n = len(next(iter(cols.values()))[0])
    idx = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(NB)]
    v = np.array([[stat(*[a[i] for a in cols[L]]) for L in cols] for i in idx])   # (NB+1, n_layers)
    win = np.bincount(v[1:].argmax(1), minlength=len(cols)) / NB
    lo, hi = np.percentile(v[1:], [2.5, 97.5], axis=0)
    return {L: dict(value=v[0, j], lo=lo[j], hi=hi[j], p_peak=win[j]) for j, L in enumerate(cols)}


def row_table(d, unit_obj, e, cond="R"):
    """One unit's rows at one layer: d_logp, headroom, d_pacc vs baseline, indexed by clip_id."""
    g = d[d.object == unit_obj].fillna({"logp_alt": -np.inf})   # nouns: no alt word
    k = g[g.cond == cond].pivot_table(index="clip_id", columns="edit", values=["logp", "logp_alt"], aggfunc="first")
    null = g[g.cond == "null"].set_index("clip_id").logp
    pacc = lambda x: (k[("logp", x)] > k[("logp_alt", x)]).astype(float)
    return pd.DataFrame(dict(d_logp=k[("logp", e)] - k[("logp", "baseline")],
                             head=k[("logp", "baseline")] - null.reindex(k.index),
                             d_pacc=pacc(e) - pacc("baseline")))


def units_rows(task, units):
    """{unit: {L: row table}} for the main edit at each layer that has one."""
    out = {u: {} for u, _ in units}
    for L in LAYERS:
        d = pd.read_parquet(f"{A}/umt_{task}_ablation_L{L}_splice/per_row.parquet")
        for u, obj in units:
            e = main_edit(d.edit.unique(), u)
            if e:
                out[u][L] = row_table(d, obj, e)
    return out


def vtc_rows(pairs):
    """{unit: {L: d_pair_correct per home-class clip}} from the VTC per_clip files."""
    out = {f"{p}_{h}": {} for p, h in pairs}
    for L in LAYERS:
        d = pd.read_parquet(f"{A}/umt_pair_vtc_L{L}_splice/per_clip.parquet")
        d["pc"] = (d.pair_margin > 0).astype(float)
        for p, h in pairs:
            g = d[(d.objects == p) & (d.true_class == h)].pivot_table(index="clip_id", columns="arm", values="pc")
            e = main_edit(g.columns, f"{p}_{h}")
            if e:
                out[f"{p}_{h}"][L] = pd.DataFrame(dict(d=g[e] - g["batch_base"]))
    return out


def certainty(rows, cols, stat, kind):
    """Align each unit's rows across its layers (same clips) and bootstrap."""
    out = []
    for u, by_L in rows.items():
        if not by_L:
            continue
        ix = sorted(set.intersection(*[set(t.index) for t in by_L.values()]))
        res = boot({L: [t.loc[ix, c].to_numpy() for c in cols] for L, t in by_L.items()}, stat)
        out += [dict(kind=kind, unit=u, L=L, n=len(ix), **r) for L, r in res.items()]
    return pd.DataFrame(out)


if __name__ == "__main__":
    nouns = ["box", "pen", "cup", "paper", "water", "book", "phone", "ball", "toy"]
    pairs = [(p, int(h)) for p in ["45_43", "168_165", "46_5", "93_86", "94_87", "40_42", "36_37", "41_44", "0_32",
                                   "59_62", "112_118", "6_171", "14_172", "8_9", "104_105", "17_18"] for h in p.split("_")]
    res = pd.concat([
        certainty(units_rows("object", [(o, o) for o in nouns]), ["d_logp", "head"], lambda d, h: -100 * d.sum() / h.sum(), "noun_mlm"),
        certainty(units_rows("pair", [(f"{p}_{h}", f"{p}:{h}") for p, h in pairs]), ["d_pacc"], lambda d: -100 * d.mean(), "verb_mlm"),
        certainty(vtc_rows(pairs), ["d"], lambda d: -100 * d.mean(), "verb_vtc")])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)
    print(res.round(2).to_string(index=False))
