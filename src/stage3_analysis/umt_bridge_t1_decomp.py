"""T1 re-aggregation (CPU, no re-extraction): split each pair's latent differences into a
WORD part and a MATCH part, from umt_vision_text_bridge.py's saved T1 outputs.

Per clip, D = latent(template A) - latent(template B). Averaging A-clips and B-clips equally:
  word  = (mean_Aclips(D) + mean_Bclips(D)) / 2   match features cancel (sign follows truth)
  match = (mean_Aclips(D) - mean_Bclips(D)) / 2   word features cancel (sign follows the word)
Blank (null) videos have no true class, so their A-B difference is a pure word signature.
Usage: uv run python src/stage3_analysis/umt_bridge_t1_decomp.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = {
    "in_dir":  ROOT / "outputs/analysis/umt_vision_text_bridge",
    "tokens":  ROOT / "notebooks/umt_ssv2_template_tokens.json",
    "layers":  (9, 10),
    "pos":     ("cls", "verb", "diff"),
    "top_n":   10,
}


def oriented_diffs(lat, g, p):
    """(n, C) A-minus-B latent differences for one pair's clips at position p.
    lat rows are [true, partner]; flip the sign on B-clips to get template A - template B."""
    X = lat[g.index.to_numpy(), :, p].astype(np.float32)          # (n, 2, C)
    sign = np.where(g.true_class.to_numpy() == g.a.iloc[0], 1.0, -1.0)[:, None]
    return (X[:, 0] - X[:, 1]) * sign, sign[:, 0] > 0              # D (n, C), is_A (n,)


def decompose(D, is_a):
    """Equal-weight A/B halves -> word (shared sign) and match (truth-following) vectors."""
    m_a, m_b = D[is_a].mean(0), D[~is_a].mean(0)
    return (m_a + m_b) / 2, (m_a - m_b) / 2


def agreement(D, is_a, f, vec, kind):
    """For feature f: (active fraction, sign agreement among active clips). Word: sign(D)
    should equal sign(word[f]) on every clip; match: equal on A-clips, opposite on B-clips."""
    d = D[:, f]
    active = d != 0
    want = np.sign(vec[f]) * (np.where(is_a, 1.0, -1.0) if kind == "match" else 1.0)
    agree = (np.sign(d) == want)[active]
    return float(active.mean()), float(agree.mean()) if agree.size else float("nan")


def null_word(null, li, p):
    """Blank-video A-B signature (mean over the 18 nulls): pure word part, no truth."""
    return (null[:, li, 0, p] - null[:, li, 1, p]).astype(np.float32).mean(0)


def summarise(D, is_a, nw, top_n):
    """Summary row + the top word/match feature ids for one (pair, layer, position)."""
    word, match = decompose(D, is_a)
    tw, tm = np.argsort(-np.abs(word))[:top_n], np.argsort(-np.abs(match))[:top_n]
    tn = set(np.argsort(-np.abs(nw))[:top_n])
    ew, em = float((word ** 2).sum()), float((match ** 2).sum())
    wa = [agreement(D, is_a, f, word, "word") for f in tw]
    ma = [agreement(D, is_a, f, match, "match") for f in tm]
    row = dict(n_a=int(is_a.sum()), n_b=int((~is_a).sum()),
               word_energy_share=ew / (ew + em),
               word_top_abs=float(np.abs(word[tw]).mean()), match_top_abs=float(np.abs(match[tm]).mean()),
               word_active=np.mean([a for a, _ in wa]), word_sign_agree=np.nanmean([s for _, s in wa]),
               match_active=np.mean([a for a, _ in ma]), match_sign_agree=np.nanmean([s for _, s in ma]),
               word_vs_null_corr=float(np.corrcoef(word, nw)[0, 1]),
               word_vs_null_top_overlap=len(set(tw) & tn) / top_n,
               match_null_abs=float(np.abs(nw[tm]).mean()))
    return row, word, match, tw, tm


def top_rows(key, L, pos, kind, ids, vec, prof, texts):
    """Readable rows: feature, its word/match value, and its 3 most-activating templates."""
    return [dict(pair=key, layer=L, pos=pos, kind=kind, feature_idx=int(f), value=float(vec[f]),
                 top_templates="; ".join(f"{texts[i]} ({prof[i, f]:.1f})"
                                         for i in np.argsort(-prof[:, f])[:3])) for f in ids]


def main():
    d = CFG["in_dir"]
    df = pd.read_parquet(d / "per_clip_t1.parquet")
    lats, nulls, profs = np.load(d / "latents_t1.npz"), np.load(d / "null_latents.npz"), np.load(d / "template_profiles.npz")
    texts = [r["text"] for r in json.load(open(CFG["tokens"]))]
    summ, tops = [], []
    for key, g in df.groupby("pair", sort=False):
        a, b = int(g.a.iloc[0]), int(g.b.iloc[0])
        for li, L in enumerate(CFG["layers"]):
            lat, prof = lats[f"l{L}"], profs[f"l{L}"].astype(np.float32)
            for p, pos in enumerate(CFG["pos"]):
                D, is_a = oriented_diffs(lat, g, p)
                row, word, match, tw, tm = summarise(D, is_a, null_word(nulls[f"{a}_{b}"], li, p), CFG["top_n"])
                summ.append(dict(pair=key, layer=L, pos=pos, **row))
                tops += top_rows(key, L, pos, "word", tw, word, prof, texts)
                tops += top_rows(key, L, pos, "match", tm, match, prof, texts)
    pd.DataFrame(summ).to_csv(d / "t1_decomp_summary.csv", index=False)
    pd.DataFrame(tops).to_csv(d / "t1_decomp_top.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(summ).drop(columns=["n_a", "n_b"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
