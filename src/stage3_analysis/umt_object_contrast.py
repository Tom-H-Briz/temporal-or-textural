"""UMT object contrast, any vision layer (03/11/26): per object (box / pen / cup), 500 val clips
whose caption names it once vs 500 clips without it from the same classes. Extraction stores
per-clip latent summaries (mean / max / top-1% mean over tokens) under R / C / Rev, and on
object clips the attribution to the masked noun (grad x z on log p(true noun), R, the contrast
runs' x = xL + decode(z - z0) path). Selection is within-class; the ablation test set is every
other object clip (umt_object_ablation.py reads sets.csv / features.csv / controls.csv).
Modes: sample (CPU) | extract (GPU) | select (CPU) | controls (CPU) | all
Usage: LAYER=7 MODE=all uv run python src/stage3_analysis/umt_object_contrast.py
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
L = int(os.environ.get("LAYER", 9))
A = ROOT / "outputs" / "analysis"
CFG = {
    "layer": L, "sae_k": 64, "job_label": "7ep", "n_lat": 6144, "n_tok": 2352,
    "objects": ("box", "pen", "cup"), "n_per_side": 500, "seed": 0,
    "summaries": ("mean", "max", "top"), "top_frac": 0.01,     # top = mean of the top 1% tokens
    "conds": ("R", "C", "Rev"), "max_len": 25,
    "min_within": 0.7, "n_feat": 3,      # stratified (same-class pairs) AUROC bar, up to 3 passers
    "ctrl_band": 0.1,                                          # controls: |AUROC - 0.5| <= band, all objects
    "max_clips": int(os.environ.get("MAX_CLIPS", 0)),          # local CPU checks only
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "refit_dir": A / "umt_mlm_refit",
    "sets_dir":  A / "umt_object_contrast",                    # layer-independent clip sets
    "out_dir":   A / f"umt_object_contrast_L{L}",
    "device": "cuda" if torch.cuda.is_available() else "cpu", "num_workers": 4,
    # TASK=pair: class A vs class B of each opposite pair (key word masked), same pipeline
    "task": os.environ.get("TASK", "object"),
    "pairs": {"45_43": ("up", "down"), "168_165": ("upwards", "downwards"), "46_5": ("opening", "closing"),
              "93_86": ("pushing", "pulling"), "94_87": ("pushing", "pulling")},
    "sel_frac": 0.5,                                           # pair clips to selection, per class
}
if CFG["task"] == "pair":                                      # own sets and outputs, never mixed
    CFG.update(objects=tuple(CFG["pairs"]), sets_dir=A / "umt_pair_contrast",
               out_dir=A / f"umt_pair_contrast_L{L}")


def run_sample(cfg):
    """sets.csv: per object 500 pos (single-mention rows) + class-matched neg (no mention of it);
    test = object rows outside every selection set. Layer-independent, seeded, written once."""
    from umt_mlm_refit import get_tokenizer, umt_pre_text
    from umt_object_ablation import CFG as ACFG, object_rows
    rows, skips = object_rows(dict(ACFG, objects=cfg["objects"]), get_tokenizer(), umt_pre_text())
    val = pd.DataFrame(json.load(open(cfg["validation_path"])))
    rng, out = np.random.default_rng(cfg["seed"]), []
    val["true_class"] = val.template.map(template_class(cfg))
    for o in cfg["objects"]:
        mention = set(rows[rows.object == o].clip_id) | set(skips[skips.object == o].clip_id)
        pos = rows[rows.object == o].sample(cfg["n_per_side"], random_state=cfg["seed"])
        pool = val[~val.id.isin(mention)]
        for c, k in pos.true_class.value_counts().items():
            g = pool[pool.true_class == c]
            out += [dict(clip_id=i, object=o, role="neg", true_class=c) for i in rng.choice(g.id, min(k, len(g)), replace=False)]
        out += [dict(r, role="pos") for r in pos.to_dict("records")]
    sel = {r["clip_id"] for r in out}
    out += [dict(r, role="test") for r in rows[~rows.clip_id.isin(sel)].to_dict("records")]
    return pd.DataFrame(out)


def run_sample_pairs(cfg):
    """sets.csv for TASK=pair: one row per class-A / class-B val clip whose caption has its key
    word exactly once (mask pos, true / alt word ids); per class sel_frac to "sel", rest "test"."""
    from umt_mlm_refit import encode, get_tokenizer, umt_pre_text
    tok, pre, cls_of, out = get_tokenizer(), umt_pre_text(), template_class(cfg), []
    val = json.load(open(cfg["validation_path"]))
    for key, words in cfg["pairs"].items():
        ab, ids = [int(x) for x in key.split("_")], tok.convert_tokens_to_ids(list(words))
        for c, w, alt in ((ab[0], ids[0], ids[1]), (ab[1], ids[1], ids[0])):
            rows = []
            for v in (v for v in val if cls_of(v["template"]) == c):
                enc, _ = encode(tok, pre(v["label"]), cfg["max_len"])
                hits = [p for p in range(1, len(enc)) if enc[p] == w]
                if len(hits) == 1:
                    rows.append(dict(clip_id=v["id"], object=key, true_class=c, pos=hits[0], true_id=w, alt_id=alt))
            r = pd.DataFrame(rows).sample(frac=1, random_state=cfg["seed"])
            n = int(round(cfg["sel_frac"] * len(r)))
            out += [r.iloc[:n].assign(role="sel"), r.iloc[n:].assign(role="test")]
    return pd.concat(out).to_dict("records")


def template_class(cfg):
    """{template: class id}, bracket-free lower-case keys as labels.json (object_rows' rule)."""
    norm = lambda s: s.lower().replace("[", "").replace("]", "")
    key2id = {norm(k): int(v) for k, v in json.load(open(cfg["labels_path"])).items()}
    return lambda t: key2id[norm(t)]


def write_sets(cfg):
    """run_sample -> sets.csv once (never overwritten: every layer must see the same clips)."""
    f = cfg["sets_dir"] / "sets.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"clip_id": str})
    cfg["sets_dir"].mkdir(parents=True, exist_ok=True)
    s = run_sample_pairs(cfg) if cfg["task"] == "pair" else run_sample(cfg)
    s = pd.DataFrame(s)
    s.to_csv(f, index=False)
    print(s.pivot_table(index="object", columns="role", values="clip_id", aggfunc="count"))
    return pd.read_csv(f, dtype={"clip_id": str})


def noun_attr(s, xL, z0, text, cfg):
    """Per masked row (one per object in the clip): grad x z of log p(true noun) at the mask,
    via x = xL + decode(z - z0) -> blocks L+1.. -> fusion -> refit head; summed over tokens."""
    from umt_bridge_phase3 import finish_tower
    from umt_mlm_refit import mlm_fusion
    cap, h8, add, true_ids = text
    out = []
    for i, (p, t) in enumerate(zip(cap["pos"], true_ids)):
        z = z0.detach().clone().float().requires_grad_(True)
        vis = finish_tower(s["model"], xL + s["sae"].decode(z - z0).unsqueeze(0), layer=cfg["layer"])
        lp = s["head"](mlm_fusion(h8[i:i + 1], add[i:i + 1], vis[0].unsqueeze(0), s["w"])[0, p]).log_softmax(-1)[t]
        g = torch.autograd.grad(lp, z)[0]
        out.append((g * z.detach()).sum(0).cpu().numpy())
    return out


@torch.no_grad()
def summaries(z0, cfg):
    """(tokens, n_lat) codes -> {mean, max, top}: per-latent mean over all tokens, max, and mean
    of the top top_frac tokens (a few patches: small objects aren't averaged away)."""
    k = max(1, int(cfg["top_frac"] * z0.shape[0]))
    return dict(mean=z0.mean(0), max=z0.max(0).values, top=z0.topk(k, dim=0).values.mean(0))


def extract_cond(cfg, s, clips, pos_rows, cond):
    """One condition over the selection clips -> summary arrays (+ noun attribution under R)."""
    from umt_bridge_phase3 import tower_to_layer
    from umt_mlm_feature_contrast import clip_loader
    from umt_object_ablation import text_batch
    from umt_vision_pair_contrast import apply_condition
    S = {k: np.full((len(clips), cfg["n_lat"]), np.nan, np.float16) for k in cfg["summaries"]}
    attr, seen = {}, set()
    for pv, y in clip_loader(clips, cfg, cond):
        i = int(y); c = clips.clip_id.iloc[i]
        if c in seen:                                  # decode-retry duplicate
            continue
        seen.add(c)
        xL = tower_to_layer(s["model"], apply_condition(pv, cond).to(cfg["device"]), layer=cfg["layer"])
        with torch.no_grad():
            z0 = s["sae"].encode((xL[0] - s["dim_mean"]).float())[1]
        for k, v in summaries(z0, cfg).items():
            S[k][i] = v.cpu().numpy()
        if cond == "R" and c in pos_rows:
            g = pos_rows[c]
            for o, a in zip(g.object, noun_attr(s, xL, z0, (*text_batch(cfg, s, g), g.true_id.tolist()), cfg)):
                attr[(c, o)] = a
    return S, attr, seen


def run_extract(cfg, s, sets):
    """summ_{cond}.npz (clip_id + mean/max/top arrays) per condition, attr.npz (pos rows, R)."""
    sel = sets[sets.role != "test"]
    clips = pd.DataFrame({"clip_id": sorted(sel.clip_id.unique())})
    pos = sets[(sets.role != "test") & sets.pos.notna() & sets.clip_id.isin(clips.clip_id)]   # rows with a masked word
    pos = pos.astype({"pos": int, "true_id": int})
    pos_rows = {c: g for c, g in pos.groupby("clip_id")}
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    for cond in cfg["conds"]:
        t0 = time.time()
        S, attr, seen = extract_cond(cfg, s, clips, pos_rows, cond)
        np.savez(cfg["out_dir"] / f"summ_{cond}.npz", clip_id=clips.clip_id.to_numpy(dtype="U"), **S)
        if attr:
            keys = list(attr)
            np.savez(cfg["out_dir"] / "attr.npz", clip_id=np.array([k[0] for k in keys], dtype="U"),
                     object=np.array([k[1] for k in keys], dtype="U"), attr=np.stack([attr[k] for k in keys]))
        print(f"  {cond}: {len(seen)}/{len(clips)} clips decoded, {time.time() - t0:.0f}s", flush=True)


def strat_auroc(X, y, cls):
    """Stratified AUROC per latent: only same-class (object, non-object) pairs are compared,
    pooled over classes (a class contributes n_obj x n_non pairs), so class can't help."""
    from scipy.stats import rankdata
    U, N = np.zeros(X.shape[1]), 0
    for c in np.unique(cls):
        m = cls == c
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if n1 and n0:
            U += rankdata(X[m].astype(np.float32), axis=0)[y[m]].sum(0) - n1 * (n1 + 1) / 2
            N += n1 * n0
    return U / N


def load_summ(cfg, cond, clip_ids):
    """summ_{cond}.npz rows for clip_ids (refuses if any requested clip is missing)."""
    z = np.load(cfg["out_dir"] / f"summ_{cond}.npz")
    idx = pd.Series(np.arange(len(z["clip_id"])), index=z["clip_id"])
    missing = set(clip_ids) - set(idx.index)
    assert not missing, f"summ_{cond}: {len(missing)} clips missing, e.g. {sorted(missing)[:3]}"
    return {k: z[k][idx[clip_ids].values].astype(np.float32) for k in cfg["summaries"]}


def object_data(cfg, sets, o):
    """One object's selection clips (decoded under every condition): table, labels, summaries."""
    d = sets[(sets.object == o) & (sets.role != "test")].reset_index(drop=True)
    Ss = {cd: load_summ(cfg, cd, d.clip_id.values) for cd in cfg["conds"]}
    ok = ~np.any([np.isnan(X).any(1) for S in Ss.values() for X in S.values()], axis=0)
    Ss = {cd: {k: X[ok] for k, X in S.items()} for cd, S in Ss.items()}
    return d[ok].reset_index(drop=True), Ss


def select_object(cfg, d, Ss, attr):
    """Candidates over all summaries (stratified AUROC on R, mean noun attribution on object
    clips) -> up to n_feat distinct passing latents, each under its best summary."""
    y, cls = (d.role == "pos").values, d.true_class.values
    A = np.stack([attr[(c, o)] for c, o in zip(d.clip_id[y], d.object[y])])
    rows = [dict(latent=f, summary=k, auroc=a, attr_mean=A[:, f].mean())
            for k in cfg["summaries"] for f, a in enumerate(strat_auroc(Ss["R"][k], y, cls))]
    c = pd.DataFrame(rows).sort_values("auroc", ascending=False).drop_duplicates("latent")
    c["passed"] = (c.auroc >= cfg["min_within"]) & (c.attr_mean > 0)
    return c, c[c.passed].head(cfg["n_feat"]), A


def check_features(d, Ss, A, chosen, y=None, cls=None):
    """Chosen latents: stratified AUROC under every condition (same summary), per-clip activation
    percentile (object / rest, R) and |noun attribution| percentile on object clips."""
    from scipy.stats import rankdata
    y = (d.role == "pos").values if y is None else y   # pairs: home-class clips, one stratum
    cls, out = (d.true_class.values if cls is None else cls), []
    attr_pct = 100 * rankdata(np.abs(A), axis=1) / A.shape[1]
    for r in chosen.itertuples():
        row = dict(latent=r.latent, summary=r.summary, attr_mean=r.attr_mean,
                   attr_pct=attr_pct[:, r.latent].mean())
        for cd, S in Ss.items():
            row[f"auroc_{cd}"] = strat_auroc(S[r.summary][:, [r.latent]], y, cls)[0]
        P = 100 * rankdata(Ss["R"][r.summary], axis=1)[:, r.latent] / A.shape[1]
        row["act_pct_obj"], row["act_pct_rest"] = P[y].mean(), P[~y].mean()
        out.append(row)
    return pd.DataFrame(out)


def select_pair(cfg, d, Ss, attr, key):
    """Per home class of the pair: AUROC(home vs other clips, R; one stratum) >= bar and mean
    key-word attribution on home clips > 0 -> up to n_feat latents; then the shared checks."""
    one, cands, feats = np.zeros(len(d), int), [], []
    for home in [int(x) for x in key.split("_")]:
        y = (d.true_class == home).values
        A = np.stack([attr[(c, key)] for c in d.clip_id[y]])
        rows = [dict(latent=f, summary=k, auroc=a, attr_mean=A[:, f].mean())
                for k in cfg["summaries"] for f, a in enumerate(strat_auroc(Ss["R"][k], y, one))]
        c = pd.DataFrame(rows).sort_values("auroc", ascending=False).drop_duplicates("latent")
        c["passed"] = (c.auroc >= cfg["min_within"]) & (c.attr_mean > 0)
        chosen = c[c.passed].head(cfg["n_feat"])
        cands.append(c.head(50).assign(home=home))
        feats.append(check_features(d, Ss, A, chosen, y=y, cls=one).assign(home=home, n_home=int(y.sum()), n_other=int((~y).sum())))
    return pd.concat(cands), pd.concat(feats)


def run_select(cfg, sets):
    """features.csv (chosen + checks, all objects) and candidates_{object}.csv (top 50)."""
    z = np.load(cfg["out_dir"] / "attr.npz")
    attr = {(c, o): a for c, o, a in zip(z["clip_id"], z["object"], z["attr"])}
    feats = []
    for o in cfg["objects"]:
        d, Ss = object_data(cfg, sets, o)
        if cfg["task"] == "pair":                     # per direction, home class in each row
            cand, f = select_pair(cfg, d, Ss, attr, o)
            cand.to_csv(cfg["out_dir"] / f"candidates_{o}.csv", index=False)
            print(f"{o}:\n{f.assign(object=o).round(3).to_string(index=False)}", flush=True)
            feats.append(f.assign(object=o))
            continue
        cand, chosen, A = select_object(cfg, d, Ss, attr)
        cand.head(50).to_csv(cfg["out_dir"] / f"candidates_{o}.csv", index=False)
        f = check_features(d, Ss, A, chosen).assign(object=o, n_obj=int((d.role == "pos").sum()), n_non=int((d.role == "neg").sum()))
        print(f"{o}: {len(chosen)} chosen\n{f.round(3).to_string(index=False)}", flush=True)
        feats.append(f)
    pd.concat(feats).to_csv(cfg["out_dir"] / "features.csv", index=False)


def run_controls(cfg, sets):
    """controls.csv: per chosen latent, the nearest-mass latent (mean activation, R, all
    selection clips) among those that separate no object (|stratified AUROC - 0.5| <= band)."""
    from umt_bridge_phase3 import mass_controls
    feats = pd.read_csv(cfg["out_dir"] / "features.csv")
    neutral, mass = np.ones(cfg["n_lat"], bool), []
    for o in cfg["objects"]:
        d, Ss = object_data(cfg, sets, o)
        if cfg["task"] == "pair":                     # class A vs B, one stratum
            a = strat_auroc(Ss["R"]["mean"], (d.true_class == int(o.split("_")[0])).values, np.zeros(len(d), int))
        else:
            a = strat_auroc(Ss["R"]["mean"], (d.role == "pos").values, d.true_class.values)
        neutral &= np.abs(a - 0.5) <= cfg["ctrl_band"]
        mass.append(Ss["R"]["mean"])
    mass = np.concatenate(mass).mean(0)
    sel = np.array(sorted(feats.latent.unique()))
    pool = set(np.where((mass > 0) & neutral)[0].tolist()) - set(sel.tolist())
    assert len(pool) >= len(sel), f"control pool has {len(pool)} latents for {len(sel)} features: widen ctrl_band"
    ctrl, gap = mass_controls(mass, sel, None, None, pool=pool)
    out = pd.DataFrame(dict(latent=sel, control=ctrl, mass=mass[sel], ctrl_mass=mass[ctrl], gap=gap))
    out.to_csv(cfg["out_dir"] / "controls.csv", index=False)
    print(out.round(4).to_string(index=False))


def smoke_sets(sets, n):
    """MAX_CLIPS=n: per object, n object + n non-object clips from one class that has both."""
    if "sel" in set(sets.role):                       # TASK=pair: n clips of each class per pair
        return sets[sets.role == "sel"].groupby(["object", "true_class"]).head(n)
    out = []
    for o, g in sets[sets.role != "test"].groupby("object"):
        k = g.groupby(["true_class", "role"]).size().unstack(fill_value=0).min(axis=1)
        c = g[g.true_class == k.idxmax()]
        out += [c[c.role == "pos"].head(n), c[c.role == "neg"].head(n)]
    return pd.concat(out)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode, cfg = os.environ.get("MODE", "all"), dict(CFG)
    if mode not in ("sample", "extract", "select", "controls", "all"):
        sys.exit(f"MODE={mode}: use sample | extract | select | controls | all")
    if cfg["max_clips"]:                              # local checks never overwrite the run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    print(f"mode={mode} layer={cfg['layer']} out={cfg['out_dir']} max_clips={cfg['max_clips'] or 'all'}", flush=True)
    sets = write_sets(cfg)
    if cfg["max_clips"]:
        sets = smoke_sets(sets, cfg["max_clips"])
        cfg["ctrl_band"] = 0.5                         # tiny n: AUROC is coarse, any latent may control
    if mode in ("extract", "all"):
        from umt_mlm_feature_contrast import setup
        s = dict(setup(cfg), labels={v["id"]: v["label"] for v in json.load(open(cfg["validation_path"]))})
        run_extract(cfg, s, sets)
    if mode in ("select", "all"):
        run_select(cfg, sets)
    if mode in ("controls", "all"):
        run_controls(cfg, sets)


if __name__ == "__main__":
    main()
