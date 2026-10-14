"""Where do Phase 3's pair-differentiating features rank by attribution, and what do the
heavy general features do? From the raw per-clip AP vectors (sparse_ap/) of UMT Phase 3:
  1. top-5 differential features per pair: rank + share of |contribution| among all latents
     for T_vtm, T_match (contribution = -AP/2) and the classifier's VTC DFA (all-174 run);
  2. per-pair profile of chosen features (default the ramps #6059, #2261, #5473): rank, share,
     contribution toward the true template on A and B clips, activation by class, selected?;
  3. those features across all 174 classes (VTC DFA): classes with > 1% share, top / bottom.
Usage: python src/stage3_analysis/phase3_feature_ranks.py [--features 6059 2261 5473]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from phase3_ablation_impact import CFG as ABL_CFG   # pair names

CFG = {
    "dir": ROOT / "outputs/analysis/umt_vision_text_bridge/phase3",
    "dfa": ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l9_job7ep_k64.parquet",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "pair_names": ABL_CFG["pair_names"],
    "targets": ("T_vtm", "T_match"),
    "top_n": 5,
}


def pair_arrays(d, clips, key):
    """(n, C) contribution = -AP/2 for target key, and (n, C) activation mass zsum."""
    zs = [np.load(d / "sparse_ap" / f"{c}.npz") for c in clips]
    return (np.stack([-0.5 * z[f"ap_{key}"].astype(np.float32) for z in zs]),
            np.stack([z["zsum"] for z in zs]))


def share_rank(C):
    """Per latent: mean over clips of |C_f| / sum |C| (share) and its rank (1 = largest)."""
    share = (np.abs(C) / (np.abs(C).sum(1, keepdims=True) + 1e-12)).mean(0)
    rank = np.empty(len(share), dtype=int)
    rank[np.argsort(-share)] = np.arange(1, len(share) + 1)
    return share, rank


def vtc_share_rank(dfa, a, b):
    """The same from the all-174 VTC DFA (signed per-latent vectors, own-class target),
    restricted to clips of the pair's two classes."""
    return share_rank(np.stack(dfa[dfa.class_id.isin([a, b])].signed_vec_R.values))


def pair_stats(cfg, ap, dfa, a, b):
    """One pair, all its clips: {target: (C, share, rank)}, zsum, is_a, VTC (share, rank)."""
    g = ap[(ap.a == a) & (ap.b == b)]
    ps = {"is_a": (g.true_class == a).to_numpy(), "vtc": vtc_share_rank(dfa, a, b)}
    for t in cfg["targets"]:
        C, ps["zsum"] = pair_arrays(cfg["dir"], g.clip_id, t)
        ps[t] = (C,) + share_rank(C)
    return ps


def top_rows(ps, sel, name, cfg):
    """The pair's top-n differential features (T_vtm selection list): rank and % share of
    |contribution| for each target and for VTC DFA."""
    rows = []
    for _, r in sel[sel.target == "T_vtm"].sort_values("rank").head(cfg["top_n"]).iterrows():
        f = int(r.feature)
        row = dict(pair=name, sel_rank=int(r["rank"]) + 1, feature=f, type="reversed" if r.ratio < 0 else "one-sided")
        for t in cfg["targets"]:
            row[f"{t}_rank"], row[f"{t}_share_pct"] = int(ps[t][2][f]), 100 * ps[t][1][f]
        row["vtc_rank"], row["vtc_share_pct"] = int(ps["vtc"][1][f]), 100 * ps["vtc"][0][f]
        rows.append(row)
    return rows


def profile_rows(ps, sel, name, feats):
    """Each chosen feature in this pair (T_vtm): rank, share, mean contribution toward the TRUE
    template on A clips (ca) and B clips (cb); evidence = (ca + cb)/2 helps whichever is true,
    bias_to_A = (ca - cb)/2 is a fixed push toward template A; activation by class; selected?"""
    (C, share, rank), is_a, z = ps["T_vtm"], ps["is_a"], ps["zsum"]
    rows = []
    for f in feats:
        ca, cb = float(C[is_a, f].mean()), float(C[~is_a, f].mean())
        rows.append(dict(pair=name, feature=f, vtm_rank=int(rank[f]), vtm_share_pct=100 * share[f],
                         contrib_on_A=ca, contrib_on_B=cb, evidence=(ca + cb) / 2, bias_to_A=(ca - cb) / 2,
                         act_A=float(z[is_a, f].mean()), act_B=float(z[~is_a, f].mean()),
                         selected=",".join(sel[sel.feature == f].target) or "-"))
    return rows


def class_reach(dfa, feats, labels):
    """Each feature across all 174 classes (VTC DFA, toward each clip's own class): classes where
    its mean share > 1%, mean share, and the 5 classes it pushes most toward / against."""
    V = np.stack(dfa.signed_vec_R.values)
    S = np.abs(V) / (np.abs(V).sum(1, keepdims=True) + 1e-12)
    fmt = lambda s: "; ".join(f"{labels[c][:45]} ({v:+.2f})" for c, v in s.items())
    rows = []
    for f in feats:
        t = pd.DataFrame({"cls": dfa.class_id.values, "share": S[:, f], "signed": V[:, f]}).groupby("cls").mean()
        rows.append(dict(feature=f, n_classes_over_1pct=int((t.share > 0.01).sum()),
                         mean_share_pct=100 * t.share.mean(), n_classes_positive=int((t.signed > 0).sum()),
                         top5_toward=fmt(t.signed.nlargest(5)), top5_against=fmt(t.signed.nsmallest(5))))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, nargs="+", default=[6059, 2261, 5473])
    feats, d = p.parse_args().features, CFG["dir"]
    ap, sel = pd.read_parquet(d / "per_clip_ap.parquet"), pd.read_csv(d / "pair_selection.csv")
    dfa = pd.read_parquet(CFG["dfa"])
    labels = {int(v): k for k, v in json.load(open(CFG["labels_path"])).items()}
    tops, prof = [], []
    for (a, b), name in CFG["pair_names"].items():
        ps, s = pair_stats(CFG, ap, dfa, a, b), sel[(sel.a == a) & (sel.b == b)]
        tops += top_rows(ps, s, name, CFG)
        prof += profile_rows(ps, s, name, feats)
    out = {"top5_differential_ranks": pd.DataFrame(tops), "feature_profile_by_pair": pd.DataFrame(prof),
           "feature_class_reach": class_reach(dfa, feats, labels)}
    for k, t in out.items():
        t.to_csv(d / f"{k}.csv", index=False)
        print(f"\n== {k} ==\n", t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
