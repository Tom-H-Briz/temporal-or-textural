"""UMT positional-feature selection at L5/L7/L9 (brief 051126 Part 1, CPU).

Replaces the L9 core-14 / MLM-pool route at every layer, L9 included: (1) DFA-mass
importance — top 5% of |signed_vec_R| clip share in >= 170 of 174 classes; (2) fires on
every pair+object clip under R and C (max > 0); (3) PER CLIP (user, 6 Nov; post-fix
profiles_{R,C,Rev}.npz from the ablation driver's profiles stage): rev_margin = median r(Rev,
R) - median r(Rev, R time-reversed) <= 0 -> content_timed (follows content, not position);
else median per-clip r(C, R) < 0.65 -> excluded; (4) the rest split by the CV of the
class-averaged R profile: < 0.2 near_flat, else ramp if |r(profile, frames 0..11)| >= 0.75,
else nonramp. Writes outputs/analysis/umt_layer_positional/sets_L{L}.csv (one row per
>= 170 pool feature) and checks the result against the regression table.
Usage: LAYER=5 uv run python src/stage3_analysis/umt_layer_positional_select.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

MODEL = os.environ.get("MODEL", "umt")                # umt | videomae | timesformer (same rules, 6 Nov)
TAG = {"umt": "umt", "videomae": "vm", "timesformer": "tf"}[MODEL]
DFA = {"umt": "dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_l{L}_job7ep_k64.parquet",
       "videomae": "dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_ssv2_all174_l{L}_job7ep_k64.parquet",
       "timesformer": "dfa_mass_delta_tf/dfa_mass_delta_tf_ssv2_all174_l{L}_job{L}_k64.parquet"}  # TF SAE job = layer

CFG = {
    "layer": int(os.environ.get("LAYER", 9)), "model": MODEL,
    "k_top": 307, "min_classes": 170, "cv_cut": 0.2, "ramp_cut": 0.75,
    "rev_margin_max": 0.0, "r_c_clip_min": 0.65,       # per-clip rule 3 (user, 6 Nov)
    "dfa": ROOT / "outputs/analysis" / DFA[MODEL],
    "pair_dir": ROOT / "outputs/analysis/umt_pair_contrast_L{L}",      # rule 2, UMT only
    "obj_dir": ROOT / "outputs/analysis/umt_object_contrast_L{L}",
    # VM: shuffle = C1, reversal = Rev1 (tubelet-level, in-grammar); stored under the R/C/Rev keys
    "conds": {"R": "R", "C": "C1", "Rev": "Rev1"} if MODEL == "videomae" else {"R": "R", "C": "C", "Rev": "Rev"},
    "profiles": ROOT / f"outputs/analysis/{TAG}_layer_positional_ablation_L{{L}}_splice/profiles_{{c}}.npz",
    "out_dir": ROOT / f"outputs/analysis/{TAG}_layer_positional",
    "smoke": int(os.environ.get("MAX_CLIPS", 0)) > 0,  # smoke: *_splice_smoke profiles -> sets_L{L}_smoke.csv
}
if CFG["smoke"]:
    CFG["profiles"] = Path(str(CFG["profiles"]).replace("_splice/", "_splice_smoke/"))

EXPECTED = {
    # per-clip rule 3 (6 Nov), checked against an independent scratch computation
    5: {"pool": (67, 56),
        "ramp": [2377, 3453, 3891, 5286, 5422, 5736, 5872],
        "nonramp": [3, 148, 271, 892, 3224, 3414, 5137, 5473], "near_flat": [173],
        "content_timed": [161, 531, 685, 966, 1016, 1575, 2012, 2035, 2669, 2801, 3356, 3700, 3829, 3856, 3934,
                          4020, 4183, 4297, 4502, 4623, 4638, 4651, 4963, 5122, 5387, 5762, 5927, 5975, 6122]},
    7: {"pool": (41, 37),
        "ramp": [2977, 2983, 3377, 4072, 5255, 5597, 5621, 5629, 6054, 6110],
        "nonramp": [368, 382, 391, 528, 1309, 1862, 2468, 2674, 2843, 3111, 3431, 4468, 4553, 5013, 5658],
        "near_flat": [289, 723, 1936, 2331], "content_timed": [1551, 4585, 5345, 5757, 6113]},
    9: {"pool": (36, 34),
        "ramp": [1540, 1707, 2261, 2417, 3623, 3948, 4627, 5043, 5473, 5825, 6059],
        "nonramp": [113, 541, 716, 813, 1063, 1282, 1347, 1694, 2254, 2383, 2629, 3034, 3419, 4301, 4788, 6052],
        "near_flat": [308, 4260, 4988], "content_timed": [65, 6127]},
}


def mass_pool(cfg):
    """Rule 1: per clip share = |signed_vec_R| / sum|.|, averaged per class (174 x 6144);
    top 5% (k = 307) per class, keep latents in the top 5% of >= 170 classes. Also returns
    n_classes_top5 and vtc_rank (rank of the share averaged over all clips, 1 = heaviest)."""
    df = pd.read_parquet(str(layer_path(cfg, "dfa")))
    V = np.stack([np.asarray(v, dtype=np.float64) for v in df.signed_vec_R])
    share = np.abs(V) / np.abs(V).sum(1, keepdims=True)
    cls = df.class_id.to_numpy()
    means = np.stack([share[cls == c].mean(0) for c in np.unique(cls)])
    n_top = np.bincount(np.argsort(-means, axis=1)[:, :cfg["k_top"]].ravel(), minlength=V.shape[1])
    rank = np.argsort(-share.mean(0)).argsort() + 1
    pool = np.where(n_top >= cfg["min_classes"])[0]
    print(f"rule 1: {len(pool)} features in the >= {cfg['min_classes']}-class pool")
    return pool, n_top, rank


def layer_path(cfg, key):
    """CFG path with {L} substituted by the layer index."""
    return Path(str(cfg[key]).format(L=cfg["layer"]))


def fires_mask(cfg):
    """Rule 2: post-fix summ_{R,C}.npz array `max`, pair + object concatenated (3,966 clips);
    keep features whose max > 0 in every clip under R and under C. VM has no such files:
    the SL-clip profiles (summed over patches) under R and C1 instead."""
    if cfg["model"] != "umt":
        f = [np.load(str(cfg["profiles"]).format(L=cfg["layer"], c=cfg["conds"][k]))["prof"] for k in ("R", "C")]
        fires = np.logical_and.reduce([(p.max(1) > 0).all(0) for p in f])
        print(f"rule 2 ({cfg['model']}, SL-clip profiles): {int(fires.sum())} features fire on every clip "
              f"under R and {cfg['conds']['C']}")
        return fires
    parts = {}
    for src in ("pair_dir", "obj_dir"):
        for cond in ("R", "C"):
            with np.load(str(layer_path(cfg, src) / f"summ_{cond}.npz")) as z:
                parts.setdefault(cond, []).append(z["max"])
    full = {c: np.concatenate(v) for c, v in parts.items()}
    fires = (full["R"] > 0).all(0) & (full["C"] > 0).all(0)
    print(f"rule 2: {int(fires.sum())} features fire on every clip under R and C")
    return fires


def clip_profiles(cfg, feats):
    """Rule 3 input: post-fix profiles_{R,C,Rev}.npz -> {cond: (n_clips, k, 12)} for feats,
    rows aligned on R's clip order (each condition is saved in its own loader order), + class ids."""
    P = {}
    for c, name in cfg["conds"].items():
        with np.load(str(cfg["profiles"]).format(L=cfg["layer"], c=name)) as z:
            P[c] = pd.Series(z["clip_id"]), z["prof"][:, :, feats].astype(np.float64), z["class_id"]
    order = {c: pd.Series(range(len(ids)), index=ids).loc[P["R"][0]].to_numpy() for c, (ids, _, _) in P.items()}
    out = {c: p[order[c]].transpose(0, 2, 1) for c, (_, p, _) in P.items()}
    print(f"rule 3 input: {len(order['R'])} clips x {len(feats)} features x {out['R'].shape[-1]} positions")
    return out, P["R"][2]


def pearson_rows(A, B):
    """Pearson r of corresponding rows along the last axis; NaN where either row is constant."""
    Am, Bm = A - A.mean(-1, keepdims=True), B - B.mean(-1, keepdims=True)
    den = np.sqrt((Am * Am).sum(-1) * (Bm * Bm).sum(-1))
    return np.where(den > 0, (Am * Bm).sum(-1) / np.where(den == 0, 1.0, den), np.nan)


def clip_rules(P):
    """Rule 3 per clip: r_C = median over clips of r(C, R); rev_margin = median r(Rev, R)
    minus median r(Rev, R reversed in time) (> 0 keeps position, <= 0 follows content)."""
    R, C, V = P["R"], P["C"], P["Rev"]
    r_c = np.nanmedian(pearson_rows(C, R), axis=0)
    margin = np.nanmedian(pearson_rows(V, R), axis=0) - np.nanmedian(pearson_rows(V, R[..., ::-1]), axis=0)
    return r_c, margin


def cv_split(R, cls, cfg):
    """Rule 4 inputs: profile = class means of the clip R profiles, averaged over classes (12
    values); CV = std/mean (ddof = 1); ramp_r = r(profile, frames 0..11)."""
    prof = np.stack([R[cls == c].mean(0) for c in np.unique(cls)]).mean(0)
    cv = prof.std(1, ddof=1) / prof.mean(1)
    return cv, pearson_rows(prof, np.broadcast_to(np.arange(prof.shape[-1], dtype=float), prof.shape))


def assign_arms(r_c, margin, cv, ramp_r, cfg):
    """content_timed if rev_margin <= 0; else excluded ('') if r_C < 0.65; else near_flat /
    ramp / nonramp by CV and |ramp_r|."""
    shaped = np.where(cv < cfg["cv_cut"], "near_flat",
                      np.where(np.abs(ramp_r) >= cfg["ramp_cut"], "ramp", "nonramp"))
    return np.where(margin <= cfg["rev_margin_max"], "content_timed",
                    np.where(r_c < cfg["r_c_clip_min"], "", shaped))


def build_table(cfg):
    """All four rules -> one row per >= 170-pool feature with the brief's columns; arm is
    blank for pool features that don't fire on every clip or are excluded by rule 3."""
    pool, n_top, rank = mass_pool(cfg)
    fires = fires_mask(cfg)
    cand = pool[fires[pool]]
    P, cls = clip_profiles(cfg, cand)
    r_c, margin = clip_rules(P)
    cv, ramp_r = cv_split(P["R"], cls, cfg)
    df = pd.DataFrame(dict(feature=pool, n_classes_top5=n_top[pool], fires_all=fires[pool],
                           vtc_rank=rank[pool])).set_index("feature")
    df = df.join(pd.DataFrame(dict(r_C_clip=r_c, rev_margin=margin, cv=cv, ramp_r=ramp_r,
                                   arm=assign_arms(r_c, margin, cv, ramp_r, cfg)), index=cand))
    df = df.reset_index().assign(arm=lambda d: d.arm.fillna(""))
    print(f"arms: {df.arm.replace('', 'excluded/not firing').value_counts().to_dict()}")
    return df


def regression_check(cfg, df):
    """Compare the selected sets (and pool / fires sizes) against the brief's table; exact
    match required, differences printed feature by feature."""
    exp = EXPECTED.get(cfg["layer"]) if cfg["model"] == "umt" else None   # UMT lists only
    if exp is None:
        print(f"no regression table for L{cfg['layer']}; skipping")
        return True
    got = {a: sorted(df.loc[df.arm == a, "feature"].astype(int).tolist())
           for a in ("ramp", "nonramp", "near_flat", "content_timed")}
    ok = all(got[a] == sorted(exp.get(a, [])) for a in got)
    print(f"regression L{cfg['layer']}: pool {len(df)} vs {exp['pool'][0]}, "
          f"fires {int(df.fires_all.sum())} vs {exp['pool'][1]}")
    for a in got:
        if got[a] != sorted(exp.get(a, [])):
            print(f"  {a} MISMATCH: missing {sorted(set(exp.get(a, [])) - set(got[a]))} "
                  f"extra {sorted(set(got[a]) - set(exp.get(a, [])))}")
    print(f"regression L{cfg['layer']}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cfg = dict(CFG)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    df = build_table(cfg)
    out = cfg["out_dir"] / f"sets_L{cfg['layer']}{'_smoke' if cfg['smoke'] else ''}.csv"
    df.to_csv(out, index=False)
    ok = regression_check(cfg, df)
    print(df[["feature", "arm", "r_C_clip", "rev_margin", "cv", "ramp_r", "vtc_rank"]].to_string(index=False))
    print(f"wrote {out}  ({'regression OK' if ok else 'REGRESSION MISMATCH'})")
    if not ok:
        sys.exit(1)                                   # stop the SLURM chain before any GPU stage


if __name__ == "__main__":
    main()
