"""UMT object ablation: do the L9 object features (box #5483, cup #6011 / #5880, pen #2428;
umt_object_features.py) carry the object noun? Mask the object word in the caption, ablate
features (spliced), measure the refit MLM head's log p(true noun) at the mask.

Clips: every val clip whose caption has box / pen / cup exactly once as a single word-piece
(each object in a clip is its own masked row, so co-occurring objects are natural controls).
Edits: baseline, 4 singles, cup pair, a mass-matched control per edit; R and C; plus one
blank-video (text-only) pass per clip.
Modes: controls (CPU, local: prints control ids for CFG) | ablate (GPU; default). Smoke test:
ablate with MAX_CLIPS=3 (CPU, writes to *_smoke, never the real run). LAYER must be unset or 9.
Usage: MODE=ablate SPLICE=1 uv run python src/stage3_analysis/umt_object_ablation.py
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

CFG = {
    "layer": 9, "sae_k": 64, "job_label": "7ep",   # the contrast runs' SAE (ablation only: dose 1)
    "objects": ("box", "pen", "cup"),
    "edits": {"box_5483": [5483], "cup_6011": [6011], "cup_5880": [5880],
              "cup_pair": [6011, 5880], "pen_2428": [2428]},
    # feature -> mass-matched control: MODE=controls output, 03/11/26 (mass gaps <= 0.5%)
    "controls": {2428: 135, 5483: 564, 5880: 300, 6011: 3633},
    "conds": ("R", "C"),
    "max_len": 25,            # caption pieces, as the contrast runs
    "splice": os.environ.get("SPLICE", "1") == "1",            # project default: spliced
    "max_clips": int(os.environ.get("MAX_CLIPS", 0)),          # local CPU checks only
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "refit_dir": ROOT / "outputs/analysis/umt_mlm_refit",
    "out_dir":   Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/umt_object_ablation"))),
    "device": "cuda" if torch.cuda.is_available() else "cpu", "num_workers": 4,
}


def run_controls(cfg):
    """Mass-matched control per edited feature: nearest mean activation (13-pair clips, R)
    among latents that separate none of the objects (AUROC in [0.4, 0.6] for all three)."""
    import umt_object_features as of
    from umt_bridge_phase3 import mass_controls
    neutral = np.ones(of.CFG["n_lat"], bool)
    for obj in cfg["objects"]:                        # usable(): checked caches, same clip set
        c, Xs, _ = of.usable(dict(of.CFG, object=obj))
        X = Xs["R"]
        neutral &= np.abs(of.auroc(X, c.has_obj.values) - 0.5) <= 0.1
    mass = X.mean(0)
    sel = np.array(sorted({f for fs in cfg["edits"].values() for f in fs}))
    pool = set(np.where((mass > 0) & neutral)[0].tolist()) - set(sel.tolist())
    ctrl, gap = mass_controls(mass, sel, None, None, pool=pool)
    for f, c, g in zip(sel, ctrl, gap):
        print(f"  {f} -> {c}  mass {mass[f]:.4f} vs {mass[c]:.4f} (gap {g:.1%})")
    print("controls =", dict(zip(sel.tolist(), ctrl.tolist())))


def object_rows(cfg, tok, pre_text):
    """One row per (val clip, object in its placeholders): masked position + true word-piece.
    Skipped (reason kept): object word not a single piece in the caption, or 2+ mentions."""
    from umt_mlm_refit import encode
    key2id = {k.lower().replace("[", "").replace("]", ""): int(v)
              for k, v in json.load(open(cfg["labels_path"])).items()}
    vocab = tok.get_vocab()
    ids = {o: {vocab[w] for w in (o, o + "s", o + "es") if w in vocab} for o in cfg["objects"]}
    rows, skips = [], []
    for v in json.load(open(cfg["validation_path"])):
        enc, _ = encode(tok, pre_text(v["label"]), cfg["max_len"])
        cls = key2id[v["template"].lower().replace("[", "").replace("]", "")]
        for o in cfg["objects"]:
            if not any(re.search(rf"\b{o}(es|s)?\b", p.lower()) for p in v["placeholders"]):
                continue
            hits = [p for p in range(1, len(enc)) if enc[p] in ids[o]]
            rec = dict(clip_id=v["id"], true_class=cls, object=o)
            (rows if len(hits) == 1 else skips).append(
                dict(rec, pos=hits[0], true_id=enc[hits[0]]) if len(hits) == 1
                else dict(rec, reason="repeated" if hits else "missing"))
    return pd.DataFrame(rows), pd.DataFrame(skips)


def edit_list(cfg):
    """[(name, feats)]: baseline, each edit, and its mass-matched control (same feature count)."""
    out = [("baseline", [])]
    for name, feats in cfg["edits"].items():
        out += [(name, feats), (f"ctrl_{name}", [cfg["controls"][f] for f in feats])]
    return out


def measures(logits, g, cond, edit):
    """Per masked row: log p(true noun), rank-1 hit, top-1 id."""
    lp = logits.float().log_softmax(-1)
    tr = torch.tensor(g.true_id.values, device=lp.device)
    lpt = lp.gather(1, tr[:, None]).squeeze(1).cpu().numpy()
    top = logits.argmax(-1).cpu().numpy()
    return [dict(clip_id=r.clip_id, object=r.object, true_class=r.true_class, cond=cond, edit=edit,
                 logp=float(lpt[i]), top1=bool(top[i] == r.true_id), top1_id=int(top[i]))
            for i, r in enumerate(g.itertuples())]


def text_batch(cfg, s, g):
    """One clip's masked captions (one row per object in g) -> (cap, h8, add) for the fusion."""
    from umt_mlm_refit import encode, mlm_text
    enc, _ = encode(s["tok"], s["pre_text"](s["labels"][g.clip_id.iloc[0]]), cfg["max_len"])
    ids = torch.tensor([[*enc[:p], s["tok"].mask_token_id, *enc[p + 1:]] for p in g.pos],
                       device=cfg["device"])
    with torch.no_grad():
        h8, add = mlm_text(ids, torch.ones_like(ids), s["w"])
    return dict(pos=g.pos.tolist()), h8, add


@torch.no_grad()
def null_logits(s, cap, h8, add):
    """Blank video (one zero vision token, the refit's null convention) -> logits at the masks."""
    from umt_mlm_refit import mlm_fusion
    vis = torch.zeros(1, 1, 768, device=h8.device, dtype=h8.dtype)
    h = mlm_fusion(h8, add, vis, s["w"])
    return s["head"](h[torch.arange(len(cap["pos"]), device=h.device), torch.tensor(cap["pos"], device=h.device)])


def run_ablate(cfg, s, rows):
    """R and C x 11 edits per clip (+ one blank-video pass) -> per_row.parquet."""
    from umt_bridge_phase3 import tower_to_layer
    from umt_mlm_feature_contrast import clip_loader
    from umt_mlm_group_ablation import edit_logits
    clips = rows.drop_duplicates("clip_id")[["clip_id"]].reset_index(drop=True)
    by_clip, edits, recs, D = dict(tuple(rows.groupby("clip_id"))), edit_list(cfg), [], s["sae"].get_dictionary()
    texts = {c: text_batch(cfg, s, by_clip[c]) for c in clips.clip_id}
    for c in clips.clip_id:
        recs += measures(null_logits(s, *texts[c]), by_clip[c], "null", "baseline")
    save = lambda: pd.DataFrame(recs).to_parquet(cfg["out_dir"] / "per_row.parquet", index=False)
    save()                                            # after each pass: a crash keeps what is done
    for cond in cfg["conds"]:
        seen, t0 = set(), time.time()
        for pv, y in clip_loader(clips, cfg, cond):
            c = clips.clip_id.iloc[int(y)]
            if c in seen:
                continue
            seen.add(c)
            x9 = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=cfg["layer"])
            with torch.no_grad():
                z0 = s["sae"].encode((x9[0] - s["dim_mean"]).float())[1]
            recs += [r for name, f in edits for r in measures(
                edit_logits(s, *texts[c][1:], texts[c][0], x9, z0, D, f, 1.0), by_clip[c], cond, name)]
        print(f"  {cond}: {len(seen)}/{len(clips)} clips, {time.time() - t0:.0f}s", flush=True)
        save()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode, cfg = os.environ.get("MODE", "ablate"), dict(CFG)
    if mode not in ("controls", "ablate"):            # nothing else may fall through to a full run
        sys.exit(f"MODE={mode}: use controls | ablate (smoke test = ablate with MAX_CLIPS=3)")
    if mode == "controls":
        return run_controls(cfg)
    if os.environ.get("LAYER", str(cfg["layer"])) != str(cfg["layer"]):   # finish_tower reads it
        sys.exit(f"LAYER={os.environ['LAYER']} in the environment but this run is L{cfg['layer']}: "
                 "unset LAYER (edit_logits' finish_tower falls back to it)")
    if cfg["splice"]:
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_splice")
    if cfg["max_clips"]:                              # local checks never overwrite the run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    from umt_mlm_feature_contrast import setup
    s = dict(setup(cfg), splice=cfg["splice"],
             labels={v["id"]: v["label"] for v in json.load(open(cfg["validation_path"]))})
    rows, skips = object_rows(cfg, s["tok"], s["pre_text"])
    skips.to_csv(cfg["out_dir"] / "skipped_rows.csv", index=False)
    if cfg["max_clips"]:
        rows = rows[rows.clip_id.isin(rows.groupby("object").clip_id.first())]   # 1 clip / object
        rows = rows[rows.clip_id.isin(rows.clip_id.unique()[:cfg["max_clips"]])]
    print(f"mode={mode} splice={cfg['splice']} out={cfg['out_dir']} rows={len(rows)} "
          f"clips={rows.clip_id.nunique()} skipped={len(skips)}", flush=True)
    run_ablate(cfg, s, rows)


if __name__ == "__main__":
    main()
