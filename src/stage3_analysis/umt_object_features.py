"""UMT object features: which L9 SAE latents separate clips showing an object (e.g. box) from
clips without it, over the 13 text-neutral pairs' clips, from the saved contrast sparse files.

Per clip: mean z over its 2352 tokens per latent (R / C / Rev). Select on R by box-vs-rest
AUROC, then keep latents that still separate within classes (class check, design option c).
Usage: OBJECT=box uv run python src/stage3_analysis/umt_object_features.py
"""
import json, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "outputs" / "analysis"
CFG = dict(object=os.environ.get("OBJECT", "box"), n_lat=6144, n_tok=2352, conds=("R", "C", "Rev"),
           runs=("umt_mlm_feature_contrast", "umt_mlm_feature_contrast_pushpull",
                 "umt_mlm_feature_contrast_tn_a", "umt_mlm_feature_contrast_tn_b"),
           skip_pairs=((37, 36), (42, 40), (44, 41)), top_k=50, min_auroc=0.7,
           min_per_side=5, min_within=0.7, out_dir=A / "umt_object_features")


def clip_table(cfg):
    """One row per unique clip: clip_id, true_class, run dir holding its sparse files, has_obj."""
    rows = []
    for run in cfg["runs"]:
        p = pd.read_parquet(A / run / "per_clip.parquet", columns=["clip_id", "a", "b", "true_class"])
        p = p[[(a, b) not in cfg["skip_pairs"] for a, b in zip(p.a, p.b)]]
        rows.append(p.drop_duplicates("clip_id").assign(run=run))
    c = pd.concat(rows).drop_duplicates("clip_id").reset_index(drop=True)
    val = {d["id"]: d for d in json.load(open(ROOT / "data/ssv2/labels/validation.json"))}
    word = re.compile(rf"\b{cfg['object']}(es|s)?\b")      # any box: "cardboard box", "boxes"
    c["has_obj"] = [any(word.search(x.lower()) for x in val[str(i)]["placeholders"]) for i in c.clip_id]
    return c


def clip_means(c, cond, cfg, field="z"):
    """(clips, n_lat) mean of field (z or dfa) over tokens for one condition; cached; NaN row if
    file missing. dfa is the clip's own pair decision (T_ab), from the run it was read from."""
    cache = cfg["out_dir"] / (f"clip_means_{cond}.npy" if field == "z" else f"clip_dfa_{cond}.npy")
    ids_file = cache.with_suffix(".ids.npy")          # row order the cache was built with
    if cache.exists():
        ids = np.load(ids_file) if ids_file.exists() else None
        X = np.load(cache)
        if ids is None or not np.array_equal(ids, c.clip_id.to_numpy(dtype="U")) \
                or X.shape[1] != cfg["n_lat"]:
            raise RuntimeError(f"{cache.name} doesn't match the current clip table / n_lat "
                               f"(runs, skip_pairs or n_lat changed?): delete {cache.name} and rebuild")
        return X
    X = np.full((len(c), cfg["n_lat"]), np.nan, dtype=np.float32)
    for i, (cid, run) in enumerate(zip(c.clip_id, c.run)):
        f = A / run / "sparse" / cond / f"{cid}.npz"
        if f.exists():
            z = np.load(f)
            X[i] = np.bincount(z["lat"], weights=z[field].astype(np.float32), minlength=cfg["n_lat"]) / cfg["n_tok"]
    np.save(cache, X)
    np.save(ids_file, c.clip_id.to_numpy(dtype="U"))
    return X


def usable(cfg):
    """Clip table + per-condition activation means + |DFA| (R), restricted to clips present in
    every condition and in DFA, so main and check always use the same clip set."""
    c = clip_table(cfg)
    Xs = {cd: clip_means(c, cd, cfg) for cd in cfg["conds"]}
    D = np.abs(clip_means(c, "R", cfg, field="dfa"))
    ok = ~np.any([np.isnan(X).any(1) for X in [*Xs.values(), D]], axis=0)
    print(f"usable clips: {ok.sum()} / {len(c)}")
    return c[ok].reset_index(drop=True), {cd: X[ok] for cd, X in Xs.items()}, D[ok]


def auroc(X, y):
    """Per-latent AUROC of X (clips, lat) for y=True vs False (ties averaged)."""
    r = rankdata(X, axis=0)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_class(X, c, cand, cfg):
    """Per-class AUROC of candidate latents, over classes with >= min_per_side clips each side."""
    out = {}
    for cls, g in c.groupby("true_class"):
        y = g.has_obj.values
        if min(y.sum(), (~y).sum()) >= cfg["min_per_side"]:
            out[cls] = auroc(X[g.index][:, cand], y)
    return pd.DataFrame(out, index=cand)          # rows: latents, cols: classes


def main(cfg):
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    c, Xs, _ = usable(cfg)
    X, y = Xs["R"], c.has_obj.values
    print(f"{cfg['object']}: {y.sum()} / {len(c)} clips; by class:\n{c.groupby('true_class').has_obj.agg(['sum', 'size']).T}")
    au = auroc(X, y)
    cand = [i for i in np.argsort(-au)[:cfg["top_k"]] if au[i] >= cfg["min_auroc"]]
    w = within_class(X, c, cand, cfg)
    res = pd.DataFrame(dict(latent=cand, auroc=au[cand], within_mean=w.mean(axis=1).values,
                            within_min=w.min(axis=1).values, n_classes=w.notna().sum(axis=1).values))
    res["passed"] = res.within_mean >= cfg["min_within"]
    pd.concat([res.set_index("latent"), w.add_prefix("cls_")], axis=1).rename_axis("latent") \
        .reset_index().to_csv(cfg["out_dir"] / f"{cfg['object']}_candidates.csv", index=False)
    print(res.round(3).to_string(index=False))


def pct(X, lats):
    """Per clip, each latent's percentile among all n_lat latents (100 = largest), then per latent."""
    return 100 * rankdata(X, axis=1)[:, lats] / X.shape[1]


def check(cfg, lats):
    """Shuffle / reverse check on chosen latents + per-clip activation and |DFA| percentiles."""
    c, Xs, D = usable(cfg)
    y = c.has_obj.values
    out = pd.DataFrame(index=pd.Index(lats, name="latent"))
    for cd, X in Xs.items():
        out[f"auroc_{cd}"] = auroc(X[:, lats], y)
        out[f"within_{cd}"] = within_class(X, c, lats, cfg).mean(axis=1).values
        out[f"act_obj_{cd}"] = X[y][:, lats].mean(0)
    for nm, M in (("act", Xs["R"]), ("dfa", D)):
        P = pct(M, lats)
        out[f"{nm}_pct_obj"], out[f"{nm}_pct_rest"] = P[y].mean(0), P[~y].mean(0)
    out.reset_index().to_csv(cfg["out_dir"] / f"{cfg['object']}_check.csv", index=False)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    if os.environ.get("MODE") == "check":
        if not os.environ.get("LATENTS"):
            sys.exit("MODE=check needs LATENTS, e.g. MODE=check OBJECT=box LATENTS=5483,5762,5973")
        check(CFG, [int(x) for x in os.environ["LATENTS"].split(",")])
    else:
        main(CFG)
