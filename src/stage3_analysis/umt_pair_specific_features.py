"""UMT MLM pairs: do pairs with their own class-specific L9 features perform better? (02/11/26)

Per pair (13 text-neutral / clean MLM pairs), on R selection clips of the contrast runs:
importance = (|mean contribution on A| + |on B|) / 2 per feature (contribution = dT_ab/dz * z,
summed over tokens). Class-specific = in the top 50 by importance (machinery excluded),
per-clip activation AUROC home vs other >= 0.8, and pushing toward its home class on home clips.
Shares: machinery (the 6 shared timing features) / class-specific / the rest, of all importance.
Performance: 174-way VTC within-pair accuracy (umt_vtm_gate per-clip logits, full val) and the
MLM pair word (sign of T_ab, R). Specific features also get shuffled (C) and reversed (Rev)
AUROC and ramp r (vision_pair_ramp_check), for the detail table.
Outputs -> outputs/analysis/umt_pair_table/: pair_summary.csv, specific_features.csv.
Usage: uv run python src/stage3_analysis/umt_pair_specific_features.py   (CPU, local sparse/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
from umt_mlm_feature_contrast import feature_vecs
from vision_pair_ramp_check import position_profiles, ramp_r

A = ROOT / "outputs/analysis"
CFG = {
    "pairs": {(45, 43): "", (168, 165): "", (46, 5): "",             # contrast run suffix
              (93, 86): "_pushpull", (94, 87): "_pushpull",
              (8, 104): "_tn_a", (61, 138): "_tn_a", (89, 148): "_tn_a",
              (106, 145): "_tn_a", (112, 118): "_tn_a", (166, 167): "_tn_a",
              (10, 106): "_tn_b", (12, 112): "_tn_b"},
    # pair kind (user, 02/11): opposites; same frame, different verb; camera left/right is
    # excluded — UMT trains with RandomHorizontalFlip (multi_modality/dataset/__init__.py:57)
    "kind": {**{p: "opposite" for p in [(45, 43), (168, 165), (46, 5), (93, 86), (94, 87), (112, 118)]},
             **{p: "same_frame" for p in [(8, 104), (10, 106), (12, 112), (106, 145), (61, 138), (89, 148)]},
             (166, 167): "excluded_flip_aug"},
    "machinery": [5473, 6059, 2261, 2417, 6127, 4788],   # shared timing features (user, 02/11)
    "top_n": 50, "auroc_min": 0.8,
    "presence_rel": 0.25,                 # present on a clip: z mass > 25% of home-class mean
    "ramp_r": 0.78,                       # |r| for ramp / content-timed (core-14 clock cut)
    "n_frames": 12, "n_latents": 6144,
    "contrast": A / "umt_mlm_feature_contrast",
    "vtc": A / "umt_vtm_gate/per_clip.parquet",
    "out_dir": A / "umt_pair_table",
}


def pair_clips(cfg, a, b):
    """R selection clips of one pair from its contrast run -> (run dir, per_clip rows)."""
    d = Path(str(cfg["contrast"]) + cfg["pairs"][(a, b)])
    per = pd.read_parquet(d / "per_clip.parquet")
    per = per[(per.condition == "R") & (per.split == "selection") & (per.a == a) & (per.b == b)]
    return d, per.sort_values("clip_id").reset_index(drop=True)


def pair_arrays(d, per):
    """(clips x latents) contribution C and z mass M on R, and is_a, from the sparse files."""
    C, M = zip(*(feature_vecs(d / "sparse", "R", c) for c in per.clip_id))
    return np.stack(C), np.stack(M), (per.true_class == per.a).to_numpy()


def auroc_cols(M, ia):
    """P(z on an A clip > z on a B clip), per latent (Mann-Whitney U / nA nB, ties averaged)."""
    R = np.apply_along_axis(rankdata, 0, M)
    nA, nB = ia.sum(), (~ia).sum()
    return (R[ia].sum(0) - nA * (nA + 1) / 2) / (nA * nB)


def feature_scores(cfg, a, b, C, M, ia):
    """Per latent: home class (heavier mean z), coverage / leakage (per-clip presence), AUROC
    home vs other, mean contribution toward home on home / other clips, importance + share."""
    mA, mB = M[ia].mean(0), M[~ia].mean(0)
    home_a = mA >= mB
    thr = cfg["presence_rel"] * np.maximum(mA, mB)
    pA, pB = (M[ia] > thr).mean(0), (M[~ia] > thr).mean(0)
    au = auroc_cols(M, ia)
    cA, cB = C[ia].mean(0), C[~ia].mean(0)
    imp = (np.abs(cA) + np.abs(cB)) / 2
    return pd.DataFrame(dict(
        feature=np.arange(M.shape[1]), home=np.where(home_a, a, b),
        coverage=np.where(home_a, pA, pB), leakage=np.where(home_a, pB, pA),
        auroc=np.where(home_a, au, 1 - au),
        c_home=np.where(home_a, cA, -cB), c_other=np.where(home_a, cB, -cA),  # + = toward home
        importance=imp, share=imp / imp.sum()))


def specific_features(cfg, s):
    """Non-machinery latents ranked by importance; class-specific = top_n, AUROC >= auroc_min,
    pushing toward home on home clips. -> (ranked table, specific rows)."""
    r = s[~s.feature.isin(cfg["machinery"])].sort_values("importance", ascending=False)
    r = r.reset_index(drop=True).assign(rank=lambda t: t.index + 1)
    spec = r[(r["rank"] <= cfg["top_n"]) & (r.auroc >= cfg["auroc_min"]) & (r.c_home > 0)]
    return r, spec


def shares(cfg, s, spec):
    """Share of all importance: machinery / class-specific / everything else."""
    mach = float(s[s.feature.isin(cfg["machinery"])].share.sum())
    sp = float(spec.share.sum())
    return dict(share_machinery=mach, share_specific=sp, share_other=1 - mach - sp)


def shape_label(cfg, shuf_r, r_r):
    """rising / falling ramp (|shuffled r| >= cut), content-timed (|R r| >= cut only), flat."""
    if abs(shuf_r) >= cfg["ramp_r"]:
        return "rising" if shuf_r > 0 else "falling"
    return "content-timed" if abs(r_r) >= cfg["ramp_r"] else "flat"


def perturbation_cols(cfg, d, per, ia, a, spec):
    """Per specific feature: per-clip AUROC home vs other under C and Rev, ramp r of the
    home-class mean frame profile under C and R, and a shape label."""
    feats = spec.feature.tolist()
    pc = dict(in_dir=d, n_frames=cfg["n_frames"], n_latents=cfg["n_latents"])
    P = {c: position_profiles(feats, per.clip_id.tolist(), c, pc) for c in ("R", "C", "Rev")}
    rows = []
    for j, r in enumerate(spec.itertuples()):
        home = ia if r.home == a else ~ia
        au = {c: auroc_cols(P[c][:, [j]].sum(2), home)[0] for c in ("C", "Rev")}
        sr, rr = ramp_r(P["C"][home, j].mean(0)), ramp_r(P["R"][home, j].mean(0))
        rows.append(dict(feature=r.feature, auroc_C=au["C"], auroc_Rev=au["Rev"],
                         ramp_r_C=sr, ramp_r_R=rr, shape=shape_label(cfg, sr, rr)))
    return pd.DataFrame(rows, columns=["feature", "auroc_C", "auroc_Rev", "ramp_r_C", "ramp_r_R", "shape"])


def vision_acc(cfg):
    """174-way VTC within-pair accuracy over all val clips: on A clips logit_a > logit_b,
    and the same for B. -> {(a, b): (acc_A, acc_B, n_A, n_B)}."""
    g = pd.read_parquet(cfg["vtc"], columns=["true_class", "vtc_logits"])
    L, y = np.stack(g.vtc_logits.values), g.true_class.to_numpy()
    out = {}
    for a, b in cfg["pairs"]:
        ma, mb = y == a, y == b
        out[(a, b)] = ((L[ma, a] > L[ma, b]).mean(), (L[mb, b] > L[mb, a]).mean(),
                       int(ma.sum()), int(mb.sum()))
    return out


def mlm_acc(d, a, b):
    """MLM pair-word accuracy per class on R, all clips of the pair (sign of T_ab)."""
    p = pd.read_parquet(d / "per_clip.parquet")
    p = p[(p.condition == "R") & (p.a == a) & (p.b == b)]
    ia = p.true_class == a
    return float((p[ia].T_ab > 0).mean()), float((p[~ia].T_ab < 0).mean())


def run_pair(cfg, a, b, vis):
    """One pair -> (summary row, specific-feature rows)."""
    d, per = pair_clips(cfg, a, b)
    C, M, ia = pair_arrays(d, per)
    s = feature_scores(cfg, a, b, C, M, ia)
    ranked, spec = specific_features(cfg, s)
    det = spec.merge(perturbation_cols(cfg, d, per, ia, a, spec), on="feature")
    det.insert(0, "pair", f"{a}/{b}")
    va, vb, na, nb = vis[(a, b)]
    ma, mb = mlm_acc(d, a, b)
    row = dict(pair=f"{a}/{b}", kind=cfg["kind"][(a, b)], a=a, b=b, n_val_a=na, n_val_b=nb, n_sel_clips=len(per),
               vis_acc_a=va, vis_acc_b=vb, vis_acc_min=min(va, vb),
               mlm_acc_a=ma, mlm_acc_b=mb, mlm_acc_min=min(ma, mb),
               n_specific=len(spec), n_specific_a=int((spec.home == a).sum()),
               n_specific_b=int((spec.home == b).sum()),
               best_specific_rank=int(spec["rank"].min()) if len(spec) else np.nan,
               **shares(cfg, s, spec))
    return row, det


def main():
    cfg = CFG
    vis = vision_acc(cfg)
    rows, dets = [], []
    for a, b in cfg["pairs"]:
        row, det = run_pair(cfg, a, b, vis)
        rows.append(row)
        dets.append(det)
        print(f"{a}/{b}: {row['n_specific']} specific, vis min {row['vis_acc_min']:.2f}", flush=True)
    summ = pd.DataFrame(rows).sort_values("vis_acc_min", ascending=False)
    summ.to_csv(cfg["out_dir"] / "pair_summary.csv", index=False)
    pd.concat(dets).to_csv(cfg["out_dir"] / "specific_features.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
