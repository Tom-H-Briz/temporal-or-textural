"""Stage 1 aggregation (CPU) for umt_vision_pair_contrast.py: rank vision L9 SAE features that
separate move up (A=45) from move down (B=43), using the saved token-level sparse data.

Per clip, condition and feature: total activation, signed contrast DFA (toward up), active
frame count, and the per-frame activation-weighted centroid slope (patch rows per frame;
image rows grow downward, so an object moving UP gives a NEGATIVE row slope).
Per feature (pooled over clips):
  mean_dfa_up / mean_dfa_down  contrast DFA on up-clips / down-clips (+ = pushes toward up)
  evidence = (mean_up - mean_down)/2   helps the TRUE class (word/match split as text side)
  bias     = (mean_up + mean_down)/2   pushes the same way on every clip
  auroc    activation AUROC, up vs down clips (R)
  slope_up / slope_down   median row-centroid slope on active clips
  rev_swap = evidence(Rev)/evidence(R)   ~ -1: flips with time reversal (direction)
  c_keep   = evidence(C)/evidence(R)     survives frame shuffle?
Usage: uv run python src/stage3_analysis/umt_vision_pair_aggregate.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_M = os.environ.get("MODEL", "umt")   # umt | videomae | timesformer (as the extractor)
_PAIR = tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(","))
_SUF = "" if _PAIR == (45, 43) else f"_{_PAIR[0]}_{_PAIR[1]}"   # matches the extractor's out_dir
CFG = {
    "in_dir":      ROOT / "outputs/analysis" / (("umt_vision_pair_contrast" if _M == "umt" else f"{_M}_vision_pair_contrast") + _SUF),
    "pair":        _PAIR,
    "conditions":  ("R", "C1" if _M == "videomae" else "C", "Rev"),   # [1] = the shuffle condition
    "n_frames":    12 if _M == "umt" else 8,      # temporal positions (VM tubelets, TF/UMT frames)
    "grid": 14, "n_latents": 6144,
    "min_frames":  3,      # active frames needed to fit a centroid slope
    "top_n":       50,
}


def clip_summary(npz, cfg):
    """One clip -> (S, dfa_sum): S = per-frame sums {z, z*row, z*col}, each (frames, latents);
    dfa_sum = signed contrast DFA summed over all tokens, (latents,)."""
    F, G, C = cfg["n_frames"], cfg["grid"], cfg["n_latents"]
    lat = npz["lat"].astype(np.int64)
    z, d = npz["z"].astype(np.float32), npz["dfa"]
    if "pos" in npz:      # stored layout-agnostic position/patch (VM/TF runs, and any re-run)
        f, p = npz["pos"].astype(np.int64), npz["patch"].astype(np.int64)
    else:                 # original UMT run: position-major tokens
        tok = npz["tok"].astype(np.int64)
        f, p = tok // (G * G), tok % (G * G)
    S = {k: np.zeros((F, C), np.float32) for k in ("z", "zr", "zc")}
    np.add.at(S["z"], (f, lat), z)
    np.add.at(S["zr"], (f, lat), z * (p // G))
    np.add.at(S["zc"], (f, lat), z * (p % G))
    dfa_sum = np.zeros(C, np.float32)
    np.add.at(dfa_sum, lat, d)
    return S, dfa_sum


def centroid_slopes(S, min_frames):
    """Least-squares slope of the per-frame centroid (rows, cols; patch units per frame) for
    every latent active in >= min_frames frames; NaN otherwise. S from clip_summary."""
    m = S["z"] > 0                                            # (F, C) active frames
    t = np.arange(S["z"].shape[0], dtype=np.float32)[:, None]
    safe = np.where(m, S["z"], 1.0)
    out = []
    for key in ("zr", "zc"):
        c = np.where(m, S[key] / safe, 0.0)                    # centroid where active
        n, st, stt = m.sum(0), (m * t).sum(0), (m * t * t).sum(0)
        sc, stc = c.sum(0), (c * t).sum(0)
        den = n * stt - st ** 2
        slope = np.where((n >= min_frames) & (den > 0), (n * stc - st * sc) / np.where(den > 0, den, 1), np.nan)
        out.append(slope.astype(np.float32))
    return out[0], out[1], m.sum(0)


def load_condition(cond, clip_ids, cfg):
    """All clips of one condition -> dict of (n_clips, latents) arrays."""
    keys = ("zsum", "dfa", "row_slope", "col_slope", "n_frames")
    A = {k: np.zeros((len(clip_ids), cfg["n_latents"]), np.float32) for k in keys}
    for i, cid in enumerate(clip_ids):
        S, dfa = clip_summary(np.load(cfg["in_dir"] / "sparse" / cond / f"{cid}.npz"), cfg)
        rs, cs, nf = centroid_slopes(S, cfg["min_frames"])
        A["zsum"][i], A["dfa"][i], A["row_slope"][i], A["col_slope"][i], A["n_frames"][i] = (
            S["z"].sum(0), dfa, rs, cs, nf)
    return A


def auroc(x, is_a):
    """Per-column AUROC of x (n, C) for A-clips vs B-clips (Mann-Whitney via ranks)."""
    from scipy.stats import rankdata
    r = rankdata(x, axis=0)
    na, nb = is_a.sum(), (~is_a).sum()
    return (r[is_a].sum(0) - na * (na + 1) / 2) / (na * nb)


def split(dfa, is_a):
    """Equal-weight up/down halves -> (mean_up, mean_down, evidence, bias)."""
    mu, md = dfa[is_a].mean(0), dfa[~is_a].mean(0)
    return mu, md, (mu - md) / 2, (mu + md) / 2


def feature_table(A, is_a):
    """All latents, one row each, metrics from R plus Rev/C comparisons."""
    R = A["R"]
    mu, md, ev, bias = split(R["dfa"], is_a)
    ev_rev, ev_c = split(A["Rev"]["dfa"], is_a)[2], split(A[CFG["conditions"][1]]["dfa"], is_a)[2]
    sgn = np.where(is_a, 1.0, -1.0)[:, None]
    act = R["zsum"] > 0
    helps = np.where(act, (np.sign(R["dfa"]) * sgn) > 0, np.nan)
    safe = np.where(np.abs(ev) > 1e-9, ev, np.nan)
    with np.errstate(all="ignore"):
        return pd.DataFrame(dict(
            feature=np.arange(len(ev)), evidence=ev, bias=bias, mean_dfa_up=mu, mean_dfa_down=md,
            active_up=act[is_a].mean(0), active_down=act[~is_a].mean(0),
            auroc=auroc(R["zsum"], is_a), helps_true_rate=np.nanmean(helps, 0),
            slope_up=np.nanmedian(R["row_slope"][is_a], 0), slope_down=np.nanmedian(R["row_slope"][~is_a], 0),
            rev_swap=ev_rev / safe, c_keep=ev_c / safe))


def main():
    cfg, a = CFG, CFG["pair"][0]
    clips = pd.read_parquet(cfg["in_dir"] / "per_clip.parquet")
    r = clips[clips.condition == "R"].reset_index(drop=True)
    ids, is_a = r.clip_id.tolist(), (r.true_class == a).to_numpy()
    sane = clips.assign(up_wins=clips.margin_ab > 0, is_up=clips.true_class == a).groupby("condition").apply(
        lambda g: pd.Series(dict(two_way_acc=float((g.up_wins == g.is_up).mean()), l0=g.l0_per_token.mean())))
    print("2-way (up vs down) accuracy by condition, true labels:\n", sane.round(3))
    A = {}
    for cond in cfg["conditions"]:
        A[cond] = load_condition(cond, ids, cfg)
        np.savez_compressed(cfg["in_dir"] / f"clip_feature_arrays_{cond}.npz", clip_id=np.array(ids), **A[cond])
    ft = feature_table(A, is_a)
    ft.to_csv(cfg["in_dir"] / "feature_table.csv", index=False)
    top = ft.sort_values("evidence", ascending=False).head(cfg["top_n"])
    top.to_csv(cfg["in_dir"] / "candidates_top.csv", index=False)
    pd.set_option("display.width", 250)
    print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
