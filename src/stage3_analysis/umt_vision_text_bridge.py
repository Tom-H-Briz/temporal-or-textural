"""UMT vision->text bridge, Phase 1 (brief dfa_attention_bridge_brief_121026.md).

Modes: g1 | g0 | t1 | all.
  g1 — gradient-path correctness (umt_fusion_grad.run_g1) -> g1_report.
  g0 — 174-way VTM top-1, raw vs SAE-spliced, fused-text SAEs L9+L10 over the
       805-clip held-out split -> g0_summary.csv / g0_pairs.csv / g0_per_clip.parquet.
  t1 — targets on real clips: the 8 usable pairs, ALL val clips of those classes.
       Per clip: VTM margin (true vs partner template); fused SAE latents at L9/L10
       for CLS, first-word verb span, and the diff span (wordpieces where the two
       templates' token sequences differ — handles [something] -> [ something ]
       automatically); D1 attention comparisons (L9-vs-L10 maps, with template-swap
       and L10-vs-L11 calibrations) ride the same pass. A null-video pass (18 nulls)
       gives the real-vs-null signature overlap and per-template activation profiles
       (top-activating templates per latent).
       -> per_clip_t1.parquet + latents_t1.npz + t1_summary.csv + t1_top_latents.csv
          + d1_summary.csv + spans.json

Two-stage persistence per the brief; raw vs intervened always within the same job.
"""
import difflib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src")]
from sae import BatchTopKSAE
from stage2_sae.train_sae_umt_fused import (
    build_loader, build_split, derive_paths, splice_scores,
)
from stage3_analysis.umt_fusion_grad import run_g1
from stage3_analysis.umt_fusion_layer_scan import _fusion_layer_attn, make_frames, synthetic_videos
from stage3_analysis.umt_text_geometry import CFG as GEOM_CFG
from stage3_analysis.umt_vtm_gate import _kv, _lin, load_vtm, vision_and_vtc
from umt_wrapper import CFG as UMT_CFG, UMTClassifier
from ToT_utils import MODEL_REGISTRY, get_frame_sampler

CFG = {
    "mode":          "g1",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "max_clips":     int(os.environ.get("MAX_CLIPS", 0)),   # smoke cap
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "manifest_path":   os.environ.get("MANIFEST_PATH",
                                     str(ROOT / "outputs/manifests/manifest_ssv2_all174_cap40.json")),
    "tokens_path":   ROOT / "notebooks/umt_ssv2_template_tokens.json",
    # the 8 usable pairs (brief design choice 4): (a, b) class ids, name per geometry CFG
    "pairs": [(6, 171), (46, 5), (14, 172), (168, 165),
              (45, 43), (44, 41), (42, 40), (37, 36)],
    "sae_layers":    (9, 10),
    "sae_k":         64, "expansion": 8, "job_label": "7ep",
    "batch_size":    8, "num_workers": 4,
    "video_dir":     __import__("os").environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "out_dir":       ROOT / "outputs/analysis/umt_vision_text_bridge",
}
GEOM_NAME = {(a, b): n for a, b, n in GEOM_CFG["pairs"]}


def verb_span(text: str, tokens: list[str]) -> list[int]:
    """Indices of the first word's wordpieces (tokens[0] is [CLS]; the vendored
    tokenizer emits no [SEP]). Reconstructs by concatenating pieces minus '##'
    until they spell the lowercased first word."""
    word = text.split()[0].lower()
    span, acc = [], ""
    for i in range(1, len(tokens)):
        span.append(i)
        acc += tokens[i].replace("##", "")
        if acc == word:
            return span
    raise ValueError(f"verb span failed: {word!r} vs {tokens}")


def pair_spans(cfg) -> dict:
    """Per pair: verb/diff token spans for both members. Diff = positions outside
    the common subsequence, so [something]'s bracket pieces are handled by the diff
    itself. Asserts every pair template exists in the token dump."""
    rows = json.load(open(cfg["tokens_path"]))
    tok = {i: r["tokens"] for i, r in enumerate(rows)}
    txt = {i: r["text"] for i, r in enumerate(rows)}
    out = {}
    for a, b in cfg["pairs"]:
        d = {"name": GEOM_NAME[(a, b)], "a": a, "b": b}
        for k, c in (("a", a), ("b", b)):
            d[f"verb_{k}"] = verb_span(txt[c], tok[c])
        sm = difflib.SequenceMatcher(None, tok[a], tok[b], autojunk=False)
        ia = sorted({x for tag, i1, i2, _, _ in sm.get_opcodes() if tag != "equal"
                     for x in range(i1, i2)})
        ib = sorted({y for tag, _, _, j1, j2 in sm.get_opcodes() if tag != "equal"
                     for y in range(j1, j2)})
        d["diff_a"], d["diff_b"] = ia, ib
        out[f"{a}_{b}"] = d
    return out


def load_fused_saes(cfg, real):
    """{(layer): eval-mode BatchTopKSAE} from the job7ep best checkpoints."""
    saes = {}
    for layer in cfg["sae_layers"]:
        ckpt = torch.load(ROOT / "outputs/sae" /
                          f"sae_umt_ssv2_fusedtext_k{cfg['sae_k']}_x{cfg['expansion']}"
                          f"_l{layer}_job{cfg['job_label']}_best.pt",
                          map_location="cpu", weights_only=True)
        sae = BatchTopKSAE(input_shape=768, nb_concepts=cfg["expansion"] * 768,
                           top_k=cfg["sae_k"] * int(real.sum().item()), device=cfg["device"])
        sae.load_state_dict(ckpt["sae_state_dict"])
        sae.running_threshold = ckpt.get("running_threshold")
        saes[layer] = sae.eval()
    return saes


@torch.no_grad()
def pair_pass(vis, text, w, a, b, qmask2):
    """One clip, the pair's two templates through layers 9-11. Returns per-layer
    hidden states {9,10}, head-mean attention maps {9,10,11} (2, 2352) plus their
    per-head forms, per-frame masses {layer: (2, 12)}, and itm scores (2,)."""
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention")
             for i in (9, 10, 11)}
    sub = {"hidden": text["hidden"][[a, b]], "add_mask": text["add_mask"][[a, b]]}
    h, states, heads, m = sub["hidden"], {}, {}, sub["add_mask"]
    for i in (9, 10, 11):
        h, probs = _fusion_layer_attn(h, m, cross[i], w, i)
        if i in (9, 10):
            states[i] = h
        q = qmask2[:, None, :, None]
        heads[i] = (probs * q).sum(2) / qmask2.sum(1)[:, None, None]   # (2, 12, 2352)
    itm = _lin(h[:, 0], w, "itm_head")[:, 1]
    maps = {i: heads[i].mean(1) for i in heads}
    frames = {i: heads[i].transpose(0, 1).reshape(12, -1).sum(-1) for i in heads}
    return states, maps, heads, frames, itm


@torch.no_grad()
def encode_latents(states, saes, dim_means, sp):
    """{(layer): (2, 3, C) fp16} — rows [template a, template b], cols [CLS, verb
    span, diff span], each the mean SAE code over that template's span."""
    out = {}
    for L, sae in saes.items():
        h = states[L]
        codes = sae((h - dim_means[L]).float().view(-1, 768))[1].view(2, h.shape[1], -1)
        vecs = [codes[:, 0],
                torch.stack([codes[0, sp["verb_a"]].mean(0), codes[1, sp["verb_b"]].mean(0)]),
                torch.stack([codes[0, sp["diff_a"]].mean(0), codes[1, sp["diff_b"]].mean(0)])]
        out[L] = torch.stack(vecs, dim=1).half().cpu()
    return out


def d1_row(maps, heads, ti):
    """D1 comparisons for one clip: same-template across layers vs the query-only
    (template-swap) calibration, head-mean + max-over-heads."""
    tv = lambda p, q: float(0.5 * (p - q).abs().sum())
    per_head = [tv(heads[9][ti, hh], heads[10][ti, hh]) for hh in range(12)]
    return {"tv_l9l10_true": tv(maps[9][ti], maps[10][ti]),
            "tv_l9l10_partner": tv(maps[9][1 - ti], maps[10][1 - ti]),
            "tv_l10l11_true": tv(maps[10][ti], maps[11][ti]),
            "tv_tmpl_l9": tv(maps[9][0], maps[9][1]), "tv_tmpl_l10": tv(maps[10][0], maps[10][1]),
            "tv_l9l10_true_headmax": float(np.max(per_head))}


def t1_clips(cfg):
    """All SSv2 val clips of the 16 pair classes, sorted by clip id; [(id, class)].
    validation.json templates are bracketed; labels.json keys are not — same
    strip+lowercase convention as umt_vtm_text_cache.py's template mapping."""
    labels = json.load(open(cfg["labels_path"]))
    key2id = {k.lower().replace("[", "").replace("]", ""): int(v) for k, v in labels.items()}
    want = {c for pair in cfg["pairs"] for c in pair}
    rows = []
    for c in json.load(open(cfg["validation_path"])):
        cid = key2id.get(c["template"].lower().replace("[", "").replace("]", ""))
        if cid in want:
            rows.append((c["id"], cid))
    rows.sort()
    if cfg["max_clips"]:
        rows = rows[: cfg["max_clips"]]
    return rows


def t1_setup(cfg):
    """Model + verified text/fusion weights + SAEs + dim_means + pair machinery."""
    from stage2_sae.train_sae_umt_fused import load_model_and_text
    model, text, w, real = load_model_and_text(cfg)
    saes = load_fused_saes(cfg, real)
    dim_means = {L: torch.load(ROOT / "outputs/sae" / f"umt_ssv2_fusedtext_layer{L}_dim_mean.pt",
                               weights_only=True).to(cfg["device"]) for L in cfg["sae_layers"]}
    spans = pair_spans(cfg)
    qmask = {p: real[[p[0], p[1]]].to(cfg["device"]) for p in cfg["pairs"]}
    return model, text, w, real, saes, dim_means, spans, qmask


def run_t1_extract(cfg, setup):
    """Stage 1: per-clip rows (margins, D1) + dense fp16 latents."""
    model, text, w, real, saes, dim_means, spans, qmask = setup

    pair_of = {c: (a, b) for a, b in cfg["pairs"] for c in (a, b)}
    rows, lats, t0 = [], [], time.time()
    loader = build_loader(t1_clips(cfg), cfg, shuffle=False)
    for pixel_values, labels in loader:
        x_vis, _ = vision_and_vtc(model, pixel_values.to(cfg["device"]))
        for vis, cls in zip(x_vis, labels.tolist()):
            a, b = pair_of[cls]
            sp, ti = spans[f"{a}_{b}"], (0 if cls == a else 1)
            states, maps, heads, _, itm = pair_pass(vis, text, w, a, b, qmask[(a, b)])
            lat = encode_latents(states, saes, dim_means, sp)
            lat = {L: v[[ti, 1 - ti]] for L, v in lat.items()}   # rows -> [true, partner]
            row = dict(clip_id=None, pair=sp["name"], a=a, b=b, true_class=cls,
                       vtm_true=float(itm[ti]), vtm_partner=float(itm[1 - ti]),
                       margin=float(itm[ti] - itm[1 - ti]), **d1_row(maps, heads, ti))
            rows.append((row, {L: lat[L].numpy() for L in lat}))
        if len(rows) % 400 < 8:
            print(f"  {len(rows):,} clips  {time.time() - t0:.0f}s", flush=True)
    ids = [c for c, _ in t1_clips(cfg)]
    for (row, _), cid in zip(rows, ids):
        row["clip_id"] = cid
    return [r for r, _ in rows], {L: np.stack([l[L] for _, l in rows]) for L in cfg["sae_layers"]}


@torch.no_grad()
def run_null_pass(cfg, setup):
    """Null videos: (a) pair latents in the same (2, 3, C) layout, for the real-vs-null
    signature overlap; (b) per-template mean codes over ALL 174 templates (averaged over
    nulls), the readable activation profile for the top-latent table."""
    from stage2_sae.train_sae_umt_fused import fused_states
    from umt_wrapper import UMTProcessor
    model, text, w, real, saes, dim_means, spans, qmask = setup
    pair_lat = {k: [] for k in spans}
    prof = {L: [] for L in cfg["sae_layers"]}
    for content, seed in synthetic_videos():
        pv = UMTProcessor()(make_frames(content, seed), return_tensors="pt")["pixel_values"].to(cfg["device"])
        vis = vision_and_vtc(model, pv)[0][0]
        for k, sp in spans.items():
            states, _, _, _, _ = pair_pass(vis, text, w, sp["a"], sp["b"], qmask[(sp["a"], sp["b"])])
            lat = encode_latents(states, saes, dim_means, sp)
            pair_lat[k].append(np.stack([lat[L].numpy() for L in cfg["sae_layers"]], 0))
        for L in cfg["sae_layers"]:
            h = fused_states(vis, text, w, L)
            codes = saes[L]((h - dim_means[L]).float().view(-1, 768))[1].view(174, 25, -1)
            prof[L].append((codes * real[:, :, None]).sum(1)
                           .div(real.sum(1)[:, None]).half().cpu().numpy())
    return {k: np.stack(v) for k, v in pair_lat.items()}, {L: np.stack(v).mean(0) for L, v in prof.items()}


def _sep_stats(D):
    """D (n, C) signed deltas true-partner: top-20 by |median| + sign consistency."""
    med = np.median(D, 0)
    top = np.argsort(-np.abs(med))[:20]
    cons = (np.sign(D[:, top]) == np.sign(med[top])).mean(0)
    return med, top, float(np.abs(med[top]).mean()), float(cons.mean()), float(cons.min())


def run_t1_aggregate(cfg, rows, lats, null_lat, prof, out):
    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_clip_t1.parquet", index=False)
    np.savez_compressed(out / "latents_t1.npz", **{f"l{L}": lats[L] for L in lats})
    np.savez_compressed(out / "null_latents.npz", **null_lat)
    np.savez_compressed(out / "template_profiles.npz", **{f"l{L}": prof[L] for L in prof})
    texts = [r["text"] for r in json.load(open(cfg["tokens_path"]))]
    pos_names, sum_rows, top_rows = ("cls", "verb", "diff"), [], []
    for key, grp in df.groupby("pair", sort=False):
        idx = grp.index.to_numpy()
        a, b = int(grp.a.iloc[0]), int(grp.b.iloc[0])
        for L in cfg["sae_layers"]:
            for p, pname in enumerate(pos_names):
                gl = lats[L][idx]                                        # (n, 2, 3, C)
                D = (gl[:, 0, p] - gl[:, 1, p]).astype(np.float32)
                med, top, tad, cmn, cmx = _sep_stats(D)
                nl = null_lat[f"{a}_{b}"][:, cfg["sae_layers"].index(L)]  # (18, 2, 3, C)
                Dn = (nl[:, 0, p] - nl[:, 1, p]).astype(np.float32)
                medn, topn, _, _, _ = _sep_stats(Dn)
                jac = len(set(top) & set(topn)) / len(set(top) | set(topn))
                sum_rows.append(dict(pair=key, layer=L, pos=pname, n=len(idx),
                                     margin_pos_rate=float((grp.margin > 0).mean()),
                                     margin_med=float(grp.margin.median()),
                                     top_abs_delta=tad, consistency_mean=cmn, consistency_min=cmx,
                                     null_jaccard_top20=jac,
                                     null_top_real_consistency=float((np.sign(D[:, topn]) == np.sign(medn[topn])).mean())))
                for f in top:
                    p5 = np.argsort(-prof[L][:, f])[:5]
                    top_rows.append(dict(pair=key, layer=L, pos=pname, feature_idx=int(f),
                                         med_delta_real=float(med[f]), med_delta_null=float(medn[f]),
                                         top_templates="; ".join(f"{texts[i]} ({prof[L][i, f]:.1f})" for i in p5)))
        pd.DataFrame(sum_rows).to_csv(out / "t1_summary.csv", index=False)
        pd.DataFrame(top_rows).to_csv(out / "t1_top_latents.csv", index=False)
    df.groupby("pair")[["tv_l9l10_true", "tv_l9l10_partner", "tv_l10l11_true",
                        "tv_tmpl_l9", "tv_tmpl_l10", "tv_l9l10_true_headmax"]].median().round(4).to_csv(
        out / "d1_summary.csv")


@torch.no_grad()
def run_g0(cfg):
    """G0: 174-way VTM top-1 raw vs SAE-spliced, both fused SAEs, 805 held-out clips."""
    from stage2_sae.train_sae_umt_fused import CFG as SAE_CFG, derive_paths
    from stage3_analysis.umt_vtm_gate import mcnemar_p
    derive_paths(SAE_CFG)
    _, val = build_split(SAE_CFG)

    model, text, w, real, saes, dim_means, _, _ = t1_setup(cfg)
    loader = build_loader(val, cfg, shuffle=False)
    classes = [c for _, c in val]
    per, raw_all, spl = [], [], {L: [] for L in cfg["sae_layers"]}
    for pixel_values, _ in loader:
        x_vis, _ = vision_and_vtc(model, pixel_values.to(cfg["device"]))
        for vis in x_vis:
            r, s9 = splice_scores(vis, text, w, real, saes[9], dim_means[9], {"layer": 9})
            _, s10 = splice_scores(vis, text, w, real, saes[10], dim_means[10], {"layer": 10})
            raw_all.append(r.cpu()); spl[9].append(s9.cpu()); spl[10].append(s10.cpu())
    raw = torch.stack(raw_all).numpy()
    y = np.array(classes)
    # Stage 1 raw per-clip data before aggregation (brief two-stage rule): 174 scores each
    pd.DataFrame(dict(clip_id=[i for i, _ in val], true_class=y, raw_scores=list(raw),
                      **{f"spliced_l{L}_scores": list(torch.stack(spl[L]).numpy()) for L in cfg["sae_layers"]})
                 ).to_parquet(cfg["out_dir"] / "g0_per_clip.parquet", index=False)
    rows = [dict(layer="raw", top1=float((raw.argmax(1) == y).mean()), n=len(y))]
    for L in cfg["sae_layers"]:
        s = torch.stack(spl[L]).numpy()
        ok_r, ok_s = raw.argmax(1) == y, s.argmax(1) == y
        rows.append(dict(layer=f"spliced_l{L}", top1=float(ok_s.mean()), n=len(y),
                         n_changed=int((ok_s != ok_r).sum()),
                         n_fixed=int((ok_s & ~ok_r).sum()), n_broken=int((~ok_s & ok_r).sum()),
                         mcnemar_p=mcnemar_p(int((ok_s & ~ok_r).sum()), int((~ok_s & ok_r).sum()))))
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "g0_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def run_t1(cfg):
    out = cfg["out_dir"]
    setup = t1_setup(cfg)
    rows, lats = run_t1_extract(cfg, setup)
    print(f"t1: {len(rows)} clips extracted; null pass...")
    null_lat, prof = run_null_pass(cfg, setup)
    run_t1_aggregate(cfg, rows, lats, null_lat, prof, out)
    print(pd.read_csv(out / "t1_summary.csv").round(4).to_string(index=False))


def main():
    import os
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = dict(CFG)
    cfg["mode"] = os.environ.get("MODE", "t1")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    json.dump(pair_spans(cfg), open(cfg["out_dir"] / "spans.json", "w"), indent=1)
    print(f"mode={cfg['mode']}  device={cfg['device']}  max_clips={cfg['max_clips'] or 'all'}")
    if cfg["mode"] in ("g1", "all"):
        from stage2_sae.train_sae_umt_fused import load_model_and_text
        _, text, w, _ = load_model_and_text(cfg)
        rep = run_g1(text, w, cfg["device"])
        json.dump({k: v for k, v in rep.items() if k != "fd_rows"},
                  open(cfg["out_dir"] / "g1_report.json", "w"), indent=1)
    if cfg["mode"] in ("g0", "all"):
        run_g0(cfg)
    if cfg["mode"] in ("t1", "all"):
        run_t1(cfg)


if __name__ == "__main__":
    main()
