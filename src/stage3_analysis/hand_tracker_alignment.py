"""Which features, in any backbone/layer, look like UMT L9 hand tracker #2782? Per up/down clip (R),
every feature's map is put on a common grid (8 time bins x VM/TF centre-crop 14x14) and scored
against #2782's map: overlap = Pearson r over cells; track = corr of centred (row, col) centroid
tracks; dist = mean centroid distance (patches). Means over clips -> one CSV per MODEL x LAYER.
Usage: MODEL=videomae LAYER=5 python src/stage3_analysis/hand_tracker_alignment.py
"""
import os
from pathlib import Path

import av
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
CFG = {"model": os.environ.get("MODEL", "umt"), "layer": int(os.environ.get("LAYER", 9)),
       "template": (2782, ROOT / "outputs/analysis/umt_vision_pair_contrast/sparse/R"),  # UMT L9, R
       "video_dir": Path(os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set"))),
       "out_dir": ROOT / "outputs/analysis/hand_tracker_alignment",
       "G": 14, "bins": 8, "umt_frames": 12, "min_bins": 3, "n_lat": 6144}


def time_matrix(T, bins):
    """(bins, T): UMT frame k sits at clip time (k+.5)/T ('middle' sampler) -> bin floor(time*bins); mean per bin."""
    M = np.zeros((bins, T))
    for k in range(T):
        M[int((k + .5) / T * bins), k] = 1
    return M / M.sum(1, keepdims=True)


def col_matrix(W, H, G):
    """(G, G): VM/TF crop column (central HxH square, landscape WxH frame) as the overlap-weighted
    mean of UMT's full-width (squashed) columns. Edges in original-frame pixels."""
    u = np.linspace(0, W, G + 1)
    v = (W - H) / 2 + np.linspace(0, H, G + 1)
    M = np.clip(np.minimum(v[1:, None], u[None, 1:]) - np.maximum(v[:-1, None], u[None, :-1]), 0, None)
    return M / M.sum(1, keepdims=True)


_MAPS = {}


def load_common(path, model, W, H, cfg):
    """One clip's sparse codes -> dense (bins*G*G, n_lat) on the common grid, plus per-feature sum |DFA|.
    VM tubelet p / TF frame j already sit in bin p / j; UMT's 12 frames and full-width columns are mapped."""
    G, s = cfg["G"], np.load(path)
    if "pos" in s.files:
        pos, patch = s["pos"].astype(int), s["patch"].astype(int)
    else:                                   # older UMT files: time-major tok = frame*G*G + patch
        assert model == "umt", path
        pos, patch = np.divmod(s["tok"].astype(int), G * G)
    n_pos = cfg["umt_frames"] if model == "umt" else cfg["bins"]
    Z = sp.csr_matrix((s["z"].astype(np.float32), (pos * G * G + patch, s["lat"].astype(int))),
                      shape=(n_pos * G * G, cfg["n_lat"]))
    if model == "umt":
        if (W, H) not in _MAPS:
            S = np.kron(np.eye(G), col_matrix(W, H, G))          # patch = row*G + col
            _MAPS[(W, H)] = sp.csr_matrix(np.kron(time_matrix(n_pos, cfg["bins"]), S))
        Z = _MAPS[(W, H)] @ Z
    dfa = np.bincount(s["lat"].astype(int), np.abs(s["dfa"].astype(np.float64)), minlength=cfg["n_lat"])
    return Z.toarray(), dfa


def centroids(A, cfg):
    """(bins, n) activation-weighted centroid row and col per time bin; NaN where silent."""
    G = cfg["G"]
    A = A.reshape(cfg["bins"], G, G, -1)
    m = A.sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((A * np.arange(G)[None, :, None, None]).sum((1, 2)) / m,
                (A * np.arange(G)[None, None, :, None]).sum((1, 2)) / m)


def overlap(A, t):
    """Pearson r between template map t (cells,) and every column of A (cells, n); NaN if either silent."""
    Ac, tc = A - A.mean(0), t - t.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (tc @ Ac) / (np.sqrt((tc ** 2).sum()) * np.sqrt((Ac ** 2).sum(0)))


def track_scores(r, c, rt, ct, min_bins):
    """Template track (rt, ct: (bins,)) vs every feature (r, c: (bins, n)) over bins where both are
    active: corr of the centred, concatenated (row, col) track (needs >= min_bins), mean distance."""
    ok = ~np.isnan(r) & ~np.isnan(rt)[:, None]
    n = ok.sum(0)
    def centred(x):
        x = np.where(ok, x, 0.0)
        return np.where(ok, x - x.sum(0) / np.maximum(n, 1), 0.0)
    T = [centred(np.broadcast_to(x[:, None], r.shape)) for x in (rt, ct)]
    F = [centred(x) for x in (r, c)]
    num = sum((a * b).sum(0) for a, b in zip(T, F))
    den = np.sqrt(sum((a ** 2).sum(0) for a in T) * sum((b ** 2).sum(0) for b in F))
    d = np.where(ok, np.hypot(np.nan_to_num(r - rt[:, None]), np.nan_to_num(c - ct[:, None])), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_bins, num / den, np.nan), np.where(n > 0, d.sum(0) / n, np.nan)


def score_clip(cid, src, cfg):
    """All features of MODEL x LAYER vs the #2782 template on one clip; None if the template is silent."""
    with av.open(str(cfg["video_dir"] / f"{cid}.webm")) as v:
        W, H = v.streams.video[0].codec_context.width, v.streams.video[0].codec_context.height
    f, tdir = cfg["template"]
    t = load_common(tdir / f"{cid}.npz", "umt", W, H, cfg)[0][:, f]
    if t.sum() == 0:
        return None
    rt, ct = (x[:, 0] for x in centroids(t[:, None], cfg))
    A, dfa = load_common(src / f"{cid}.npz", cfg["model"], W, H, cfg)
    r, c = centroids(A, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc, dist = track_scores(r, c, rt, ct, cfg["min_bins"])
    return dict(overlap=overlap(A, t), track=tc, dist=dist, dfa=dfa, active=A.sum(0) > 0)


def main(cfg=CFG):
    m, L = cfg["model"], cfg["layer"]
    name = ("umt_vision_pair_contrast" if m == "umt" else f"{m}_vision_pair_contrast") + ("" if L == 9 else f"_L{L}")
    src = ROOT / "outputs/analysis" / name / "sparse/R"
    ids = sorted(p.stem for p in src.glob("*.npz"))
    acc = {k: [] for k in ("overlap", "track", "dist", "dfa", "active")}
    for cid in ids:
        out = score_clip(cid, src, cfg)
        for k in acc if out else ():
            acc[k].append(out[k])
    a = {k: np.array(v) for k, v in acc.items()}
    with np.errstate(invalid="ignore"):
        df = pd.DataFrame({k: np.nanmean(a[k], 0) for k in ("overlap", "track", "dist")})
    df.insert(0, "feature", np.arange(cfg["n_lat"]))
    df["coact"], df["dfa_share"] = a["active"].mean(0), a["dfa"].sum(0) / a["dfa"].sum()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg["out_dir"] / f"{m}_L{L}.csv", index=False)
    print(f"{m} L{L}: {len(a['active'])}/{len(ids)} clips -> {cfg['out_dir'] / f'{m}_L{L}.csv'}")


if __name__ == "__main__":
    main()
