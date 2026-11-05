"""UMT positional-feature selection at L5/L7/L9 (brief 051126 Part 1, CPU).

Replaces the L9 core-14 / MLM-pool route at every layer, L9 included: (1) DFA-mass
importance — top 5% of |signed_vec_R| clip share in >= 170 of 174 classes; (2) fires on
every pair+object clip under R and C (max > 0); (3) shape consistent with itself — median
per-class r(R profile, C profile) >= 0.9; (4) CV split of the class-averaged R frame
profile: >= 0.2 positional (ramp if |r(profile, frames 0..11)| >= 0.75, else nonramp),
< 0.2 near_flat. Writes outputs/analysis/umt_layer_positional/sets_L{L}.csv (one row per
>= 170 pool feature) and checks the result against the brief's regression table.
Usage: LAYER=5 uv run python src/stage3_analysis/umt_layer_positional_select.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

CFG = {
    "layer": int(os.environ.get("LAYER", 9)),
    "k_top": 307, "min_classes": 170, "r_min": 0.9, "cv_cut": 0.2, "ramp_cut": 0.75,
    "dfa": ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_l{L}_job7ep_k64.parquet",
    "pair_dir": ROOT / "outputs/analysis/umt_pair_contrast_L{L}",
    "obj_dir": ROOT / "outputs/analysis/umt_object_contrast_L{L}",
    "lock": ROOT / "outputs/analysis/position_lock/position_lock_umt_ssv2_l{L}_job7ep_k64.parquet",
    "out_dir": ROOT / "outputs/analysis/umt_layer_positional",
}

EXPECTED = {
    5: {"pool": (67, 56),
        "ramp": [2377, 2705, 3453, 3891, 5286, 5422, 5736, 5872],
        "nonramp": [3, 148, 271, 892, 1004, 2801, 3224, 3414, 5122, 5137, 5473],
        "near_flat": [173, 934, 1079, 2012, 2362, 3700, 3856, 3934, 4020, 4502, 5387, 5762, 5927, 5975]},
    7: {"pool": (41, 37),
        "ramp": [2977, 2983, 3377, 4072, 5255, 5597, 5621, 5629, 6054, 6110],
        "nonramp": [368, 382, 391, 528, 1309, 1862, 2468, 2674, 2843, 3111, 3431, 4468, 4553, 5013, 5658, 6113],
        "near_flat": [289, 723, 1551, 1936, 2331]},          # 2318 out: r_RC 0.8954 < 0.9
    9: {"pool": (36, 34),
        "ramp": [1540, 1707, 2261, 2417, 3623, 3948, 4627, 5043, 5473, 5825, 6059],
        "nonramp": [113, 541, 716, 813, 1063, 1282, 1347, 1694, 2254, 2383, 2606, 2629, 3034, 3419, 4301, 4788, 6052],
        "near_flat": [308, 4260, 4988]},
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
    keep features whose max > 0 in every clip under R and under C."""
    parts = {}
    for src in ("pair_dir", "obj_dir"):
        for cond in ("R", "C"):
            with np.load(str(layer_path(cfg, src) / f"summ_{cond}.npz")) as z:
                parts.setdefault(cond, []).append(z["max"])
    full = {c: np.concatenate(v) for c, v in parts.items()}
    fires = (full["R"] > 0).all(0) & (full["C"] > 0).all(0)
    print(f"rule 2: {int(fires.sum())} features fire on every clip under R and C")
    return fires


def frame_profiles(cfg):
    """Rule 3 input: position_lock parquet -> (R, C) arrays (n_classes, 6144, 12); activation
    summed over the 196 patches of a frame, one row per class_id x feature_idx."""
    pl = pd.read_parquet(str(layer_path(cfg, "lock")))
    piv = pl.pivot_table(index=["class_id", "feature_idx"], columns="frame_idx",
                         values=["mean_raw_activation_R", "mean_raw_activation_C"])
    n = pl.class_id.nunique()
    R = piv["mean_raw_activation_R"].to_numpy().reshape(n, 6144, 12)
    C = piv["mean_raw_activation_C"].to_numpy().reshape(n, 6144, 12)
    print(f"rule 3 input: {n} classes x 6144 features x 12 frames")
    return R.astype(np.float64), C.astype(np.float64)


def pearson_rows(A, B):
    """Pearson r of corresponding rows along the last axis; NaN where either row is constant."""
    Am, Bm = A - A.mean(-1, keepdims=True), B - B.mean(-1, keepdims=True)
    den = np.sqrt((Am * Am).sum(-1) * (Bm * Bm).sum(-1))
    return np.where(den > 0, (Am * Bm).sum(-1) / np.where(den == 0, 1.0, den), np.nan)


def consistency(R, C):
    """Rule 3: r_RC = median over classes of r(R profile, C profile); xclass_r = the
    cross-class consistency (median per class of r(class R profile, class-averaged R
    profile)) — computed and stored, never binds."""
    r_rc = np.median(pearson_rows(R, C), axis=0)
    avg = np.broadcast_to(R.mean(0, keepdims=True), R.shape)
    r_x = np.median(pearson_rows(R, avg), axis=0)
    n = int((r_rc >= 0.9).sum())
    print(f"rule 3: {n} features with r_RC >= 0.9")
    return r_rc, r_x


def cv_split(R, cfg):
    """Rule 4: profile = R averaged over the classes (12 values); CV = std/mean (ddof = 1).
    CV >= 0.2 positional else near_flat; within positional, ramp if |r(profile, 0..11)| >= 0.75."""
    prof = R.mean(0)
    cv = prof.std(1, ddof=1) / prof.mean(1)
    ramp_r = pearson_rows(prof, np.broadcast_to(np.arange(12.0), prof.shape))
    arm = np.where(cv < cfg["cv_cut"], "near_flat",
                   np.where(np.abs(ramp_r) >= cfg["ramp_cut"], "ramp", "nonramp"))
    for a in ("ramp", "nonramp", "near_flat"):
        print(f"rule 4: {int((arm == a).sum())} {a}")
    return cv, ramp_r, arm


def build_table(cfg):
    """All four rules -> one row per >= 170-pool feature with the brief's columns; arm is
    blank for pool features that fail the fire or consistency rule."""
    pool, n_top, rank = mass_pool(cfg)
    fires = fires_mask(cfg)
    R, C = frame_profiles(cfg)
    r_rc, r_x = consistency(R, C)
    cv, ramp_r, arm = cv_split(R, cfg)
    ok = fires[pool] & (r_rc[pool] >= cfg["r_min"])
    df = pd.DataFrame(dict(feature=pool, n_classes_top5=n_top[pool], fires_all=fires[pool],
                           r_RC=r_rc[pool], xclass_r=r_x[pool], cv=cv[pool], ramp_r=ramp_r[pool],
                           vtc_rank=rank[pool],
                           arm=np.where(ok, arm[pool], "")))
    print(f"selected: {int((df.arm != '').sum())} of {len(df)} pool features")
    return df


def regression_check(cfg, df):
    """Compare the selected sets (and pool / fires sizes) against the brief's table; exact
    match required, differences printed feature by feature."""
    exp = EXPECTED.get(cfg["layer"])
    if exp is None:
        print(f"no regression table for L{cfg['layer']}; skipping")
        return True
    got = {a: sorted(df.loc[df.arm == a, "feature"].astype(int).tolist())
           for a in ("ramp", "nonramp", "near_flat")}
    ok = all(got[a] == sorted(exp[a]) for a in got)
    print(f"regression L{cfg['layer']}: pool {len(df)} vs {exp['pool'][0]}, "
          f"fires {int(df.fires_all.sum())} vs {exp['pool'][1]}")
    for a in got:
        if got[a] != sorted(exp[a]):
            print(f"  {a} MISMATCH: missing {sorted(set(exp[a]) - set(got[a]))} "
                  f"extra {sorted(set(got[a]) - set(exp[a]))}")
    print(f"regression L{cfg['layer']}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    cfg = dict(CFG)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    df = build_table(cfg)
    out = cfg["out_dir"] / f"sets_L{cfg['layer']}.csv"
    df.to_csv(out, index=False)
    ok = regression_check(cfg, df)
    print(df[["feature", "arm", "r_RC", "cv", "ramp_r", "vtc_rank"]].to_string(index=False))
    print(f"wrote {out}  ({'regression OK' if ok else 'REGRESSION MISMATCH'})")


if __name__ == "__main__":
    main()
