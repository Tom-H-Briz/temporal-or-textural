"""Phase 3 of the vision->text bridge (findings/briefs/dfa_attention_bridge_brief_121026.md):
which vision-L9 SAE features does the text side use as evidence?

Targets per clip (true template t vs pair partner p, diff-span position):
  T_vtm    = itm(t) - itm(p)                                     (behavioural)
  T_match  = sum_f sign(match_f) [code_f(t) - code_f(p)]          (top-10 L10 match features)
  T_match9 = same with the L9 match features                      (cheap third backward)
AP_f = dT/dz_f * (-2 z_f) through the grad-enabled path block-9 residual -> blocks 10-11 ->
encoder.norm -> fusion 9-11 -> text SAE encode (ReLU codes, detached threshold mask).
The gradient is taken at the RAW residual x9 (x = x9 + decode(z - z0)), the same point the
real edits start from, so the SAE error term is held fixed rather than dropped.

Stages (STAGE env, chained by the SLURM script): g1b | ap | select | intervene | aggregate,
plus selcheck (CPU validation of the selection on the on-file VTC contrast DFA).
Selection (agreed 13/10, replaces the brief's per-clip top-|AP|): per pair and target, the
features whose A-minus-B oriented AP is heavy on one class and silent or sign-reversed on
the other (evidence > 0), chosen on clips held out from the intervention subset; each gets
one mass-matched control that contributes alike on both classes.
Interventions are raw-stream edits x' = x9 - dose * sum(z_f d_f) (dose 1 = ablate, 2 = sign
flip), never SAE reconstruction; baseline and intervened forwards run in the same process on
the same decoded tensors. Route split patches the key or value projection of fusion L9, L10,
or all three cross-attention layers (9-11) at once.
Usage: STAGE=ap MAX_CLIPS=20 uv run python src/stage3_analysis/umt_bridge_phase3.py
"""
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

CFG = {
    "stage":        os.environ.get("STAGE", "g1b"),
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "max_clips":    int(os.environ.get("MAX_CLIPS", 0)),          # smoke cap
    "g34_clips":    int(os.environ.get("G34_CLIPS", 300)),        # intervention subset
    "labels_path":    os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":      os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "tokens_path":  ROOT / "notebooks/umt_ssv2_template_tokens.json",
    "pairs":        [(6, 171), (46, 5), (14, 172), (168, 165),
                     (45, 43), (44, 41), (42, 40), (37, 36)],
    "vis_layer":    9, "sae_k": 64, "job_label": "7ep",           # vision source SAE
    "top_n_ap":     20,       # per-clip top-|AP| list, stored as a diagnostic only
    # Per-pair discriminative selection (replaces per-clip top-|AP|, agreed 13/10)
    "n_sel":        10,       # verb-evidence features per (pair, target)
    "max_other":    1 / 3,    # other class: silent (<= 1/3 of heavy side, same sign) or reversed
    "set_sizes":    (1, 5, 10),   # G4 sets top[:k]; control set = the k matched controls
    "route_layers": (9, 10, "all"),
    "batch_size":   8, "num_workers": 4,
    "out_dir":      ROOT / "outputs/analysis/umt_vision_text_bridge/phase3",
}


def load_matches(cfg):
    """{(pair_key, layer): [(feature, sign), ...]} — top-10 match features per pair from
    Phase 1's t1_decomp_top.csv (kind=match, pos=diff); sign(value) orients T_match."""
    from stage3_analysis.umt_text_geometry import CFG as GEOM_CFG
    name2pair = {n: (a, b) for a, b, n in GEOM_CFG["pairs"]}
    t = pd.read_csv(ROOT / "outputs/analysis/umt_vision_text_bridge/t1_decomp_top.csv")
    t = t[(t.kind == "match") & (t.pos == "diff")]
    out = {}
    for _, r in t.iterrows():
        a, b = name2pair[r.pair]
        out.setdefault((f"{a}_{b}", int(r.layer)), []).append(
            (int(r.feature_idx), 1.0 if r.value >= 0 else -1.0))
    return {k: v[:10] for k, v in out.items()}


def phase3_setup(cfg):
    """Model + verified fusion weights/text cache + vision SAE (trained threshold) + text
    SAEs + spans + match lists. Reuses the verified loaders; nothing no-grad is edited."""
    from huggingface_hub import hf_hub_download
    from umt_wrapper import CFG as UMT_CFG, UMTClassifier
    from umt_vtm_gate import load_vtm
    from ToT_utils import load_sae_eval, resolve_sae_checkpoint
    from stage3_analysis.umt_vision_text_bridge import load_fused_saes, pair_spans

    model = UMTClassifier.from_pretrained().to(cfg["device"]).eval()
    sd = torch.load(hf_hub_download(UMT_CFG["hf_repo"], UMT_CFG["hf_filename"]),
                    map_location="cpu", weights_only=False)
    text, w = load_vtm(sd, cfg["device"])
    res = resolve_sae_checkpoint("umt", cfg["vis_layer"], dataset_name="ssv2",
                                 sae_k=cfg["sae_k"], job_label=cfg["job_label"])
    sae = load_sae_eval(res["sae_path"], input_dim=768, tokens_per_clip=2352,
                        device=cfg["device"], sae_k=cfg["sae_k"])
    dim_mean = torch.load(res["dim_mean_path"], weights_only=True).to(cfg["device"])
    fused_cfg = dict(sae_layers=(9, 10), sae_k=64, expansion=8, job_label="7ep",
                     device=cfg["device"])
    real = torch.from_numpy(np.load(ROOT / "notebooks/umt_ssv2_template_bert_l8.npz")["mask"]).float()
    tsaes = load_fused_saes(fused_cfg, real=real)
    return {"model": model, "text": text, "w": w, "sae": sae, "dim_mean": dim_mean,
            "tsaes": tsaes, "real": real.to(cfg["device"]), "tdm": {L: torch.load(ROOT / "outputs/sae" /
                                 f"umt_ssv2_fusedtext_layer{L}_dim_mean.pt", weights_only=True
                                 ).to(cfg["device"]) for L in (9, 10)},
            "spans": pair_spans({**cfg, "pairs": cfg["pairs"], "tokens_path": cfg["tokens_path"]}),
            "matches": load_matches(cfg)}


def tower_to_layer9(model, pixel_values):
    """(B, 2352, 768) block-9 residual (the vision SAE's input), no grad: patch_embed +
    pos_embed + blocks 0..9 — exactly forward_features up to the layer_getter hook point.
    The tower lives on model.vision.encoder (same object ToT_utils' layer_getter uses)."""
    v = model.vision.encoder
    with torch.no_grad():
        x = v.patch_embed(pixel_values.permute(0, 2, 1, 3, 4).float())
        x = x + v.pos_embed.type_as(x).to(x.device).clone().detach()
        for blk in v.blocks[:10]:
            x = blk(x)
    return x


def finish_tower(model, x9):
    """Blocks 10-11 + encoder.norm, grad-enabled: the final tokens cross-attention reads."""
    v = model.vision.encoder
    for blk in v.blocks[10:]:
        x9 = blk(x9)
    return v.norm(x9)


def cross_kv(vis, w, i, route=None, alt_vis=None):
    """(k, v) heads for fusion layer i's cross-attention. route=("key"|"value", i | "all")
    substitutes alt_vis into that projection of layer i (or of every layer for "all") — the
    G4 route split; other layers/projections use vis."""
    from umt_vtm_gate import _heads, _lin
    pre = f"text_encoder.encoder.layer.{i}.crossattention"
    proj, layer = route if route is not None else (None, None)
    hit = layer in (i, "all")
    kv_vis = alt_vis if (hit and proj == "key") else vis
    vv_vis = alt_vis if (hit and proj == "value") else vis
    return (_heads(_lin(kv_vis.unsqueeze(0), w, pre + ".self.key")),
            _heads(_lin(vv_vis.unsqueeze(0), w, pre + ".self.value")))


def fusion_pair(vis, text, w, a, b, route=None, alt_vis=None):
    """The pair's two templates through grad fusion 9-11 off vis -> (states {9,10}, itm (2,)).
    Mirrors umt_vision_text_bridge.pair_pass (same layer order, same itm head)."""
    from umt_fusion_grad import fusion_layer_grad
    from umt_vtm_gate import _lin
    cross = {i: cross_kv(vis, w, i, route, alt_vis) for i in (9, 10, 11)}
    h, m = text["hidden"][[a, b]], text["add_mask"][[a, b]]
    states = {}
    for i in (9, 10, 11):
        h = fusion_layer_grad(h, m, cross[i], w, i)
        if i in (9, 10):
            states[i] = h
    itm = _lin(h[:, 0], w, "itm_head")[:, 1]
    return states, itm


def span_codes(sae, dm, h_row, span):
    """Mean fused-text SAE codes over ONE template's span tokens (h_row (25, 768)),
    differentiable a.e.: ReLU codes, detached running-threshold mask (eval encode)."""
    return sae(h_row[span] - dm)[1].mean(0)


def targets(vis_final, setup, sp, matches, ti, route=None, alt_vis=None):
    """The three Phase 3 targets from final vision tokens. ti indexes the TRUE template
    (rows are [a, b]); T_vtm = itm(true) - itm(partner); T_match(L) = sum_f sign_f
    (codes_true[f] - codes_partner[f]) over that pair-layer's top match features."""
    w, tsaes, tdm = setup["w"], setup["tsaes"], setup["tdm"]
    states, itm = fusion_pair(vis_final, setup["text"], w, sp["a"], sp["b"],
                              route=route, alt_vis=alt_vis)
    t_row, p_row = ti, 1 - ti
    out = {"T_vtm": itm[ti] - itm[1 - ti]}
    for L in (9, 10):
        fs = matches.get((f"{sp['a']}_{sp['b']}", L), [])
        key = f"T_match{L}" if L == 9 else "T_match"
        if not fs:
            out[key] = torch.zeros((), device=itm.device)
            continue
        idx = torch.tensor([f for f, _ in fs], device=itm.device)
        sgn = torch.tensor([s for _, s in fs], device=itm.device, dtype=torch.float32)
        ct = span_codes(tsaes[L], tdm[L], states[L][t_row], sp[f"diff_{'a' if ti == 0 else 'b'}"])
        cp = span_codes(tsaes[L], tdm[L], states[L][p_row], sp[f"diff_{'b' if ti == 0 else 'a'}"])
        out[key] = (sgn * (ct[idx] - cp[idx])).sum()
    return out, states


def clip_ap(x9, setup, sp, matches, ti):
    """Grad pass for one clip at the RAW residual: z (eval threshold, graph leaf) ->
    x = x9 + decode(z - z0) (equals x9 at z = z0, so the SAE error is kept) -> blocks 10-11
    -> fusion -> targets; AP(tok, f) = dz-grad * (-2 z), reduced per feature and kept
    token-level for the top features. Returns (ap_feat, ap_tok, zsum, t_base)."""
    sae, dm = setup["sae"], setup["dim_mean"]
    with torch.no_grad():
        _, z0 = sae.encode((x9[0] - dm).float())
    z = z0.detach().clone().float().requires_grad_(True)
    x = x9 + sae.decode(z - z0).unsqueeze(0)
    t, _ = targets(finish_tower(setup["model"], x)[0], setup, sp, matches, ti)   # (2352, 768)
    t_base = {k: float(v.detach()) for k, v in t.items()}
    ap_feat, ap_tok = {}, {}
    for k, T in t.items():
        g = torch.autograd.grad(T, z, retain_graph=True)[0].detach()
        tok = (g * (-2.0 * z.detach())).cpu().numpy()              # (2352, 6144) fp32
        ap_feat[k] = tok.sum(0).astype(np.float16)
        ap_tok[k] = tok
    return ap_feat, ap_tok, z0.sum(0).detach().float().cpu().numpy(), t_base


def pair_pass_ref(vis, setup, a, b):
    """Phase 1's verified no-grad pair pass (states, maps, heads, frames, itm)."""
    from stage3_analysis.umt_vision_text_bridge import pair_pass
    return pair_pass(vis, setup["text"], setup["w"], a, b, setup["real"][[a, b]])


def run_g1b(cfg, setup):
    """Check A: the extended grad path (blocks 10-11 + fusion + text-SAE encode) reproduces
    the verified no-grad path (vision_and_vtc tokens, pair_pass margins, encode_latents codes)
    on the same decoded clips. Check B: autograd vs central FD on x9, in fp64."""
    from stage2_sae.train_sae_umt_fused import build_loader
    from stage3_analysis.umt_vision_text_bridge import encode_latents, pair_spans, t1_clips
    from stage3_analysis.umt_vtm_gate import vision_and_vtc
    model, w = setup["model"], setup["w"]
    spans, matches = setup["spans"], setup["matches"]
    loader = build_loader(t1_clips({**cfg, "max_clips": 3}), cfg, shuffle=False)
    pair_of = {c: p for p in cfg["pairs"] for c in p}
    errs = {"vis": 0.0, "vtm": 0.0, "match": 0.0}
    for pv, labels in loader:
        x9 = tower_to_layer9(model, pv.to(cfg["device"]))
        with torch.no_grad():
            vis_ref = vision_and_vtc(model, pv.to(cfg["device"]))[0]
        vis_grad = finish_tower(model, x9.clone())
        errs["vis"] = max(errs["vis"], (vis_grad - vis_ref).abs().max().item())
        for vis, cls in zip(vis_ref, labels.tolist()):
            a, b = pair_of[cls]
            sp, ti = spans[f"{a}_{b}"], (0 if cls == a else 1)
            t, _ = targets(vis, setup, sp, matches, ti)
            with torch.no_grad():
                states, _, _, _, itm = pair_pass_ref(vis, setup, a, b)
                errs["vtm"] = max(errs["vtm"], abs(float(t["T_vtm"]) - float(itm[ti] - itm[1 - ti])))
                lat = encode_latents(states, setup["tsaes"], setup["tdm"], sp)
                for L in (9, 10):
                    fs = matches.get((f"{a}_{b}", L), [])
                    idx = [f for f, _ in fs]
                    ref = sum(s * float(lat[L][ti, 2, f] - lat[L][1 - ti, 2, f]) for f, s in fs)
                    key = f"T_match{L}" if L == 9 else "T_match"
                    errs["match"] = max(errs["match"], abs(float(t[key]) - ref) if idx else 0.0)
    print("G1b Check A:", {k: f"{v:.2e}" for k, v in errs.items()})
    return errs


def g1b_check_b(cfg, setup):
    """Check B on the extended path: autograd vs central finite differences w.r.t. the
    block-9 residual, fp64 (targets are O(1-10); fp32 FD roundoff would fake failures).
    Reuses umt_fusion_grad._fd_check's direction/eps conventions. Modules are deep-copied
    before .double(): nn.Module.double() is in-place and would leave the shared setup fp64."""
    from copy import deepcopy
    s64 = {"model": deepcopy(setup["model"]).double(),
           "text": {"hidden": setup["text"]["hidden"].double(),
                    "add_mask": setup["text"]["add_mask"].double()},
           "w": {k: v.double() for k, v in setup["w"].items()}, "dim_mean": setup["dim_mean"].double()}
    s64["tsaes"] = {}
    for L, sae in setup["tsaes"].items():
        # eval-mode SAEs cache a fused dictionary that is a non-leaf plain attribute: deepcopy
        # refuses it and .double() would skip it, so strip, copy, then reattach as fp64.
        fused, sae.dictionary._fused_dictionary = sae.dictionary._fused_dictionary, None
        s = deepcopy(sae).double()
        sae.dictionary._fused_dictionary = fused
        s.dictionary._fused_dictionary = fused.detach().double()
        s.running_threshold = torch.as_tensor(sae.running_threshold).double()
        s64["tsaes"][L] = s
    s64["tdm"] = {L: dm.double() for L, dm in setup["tdm"].items()}
    s64["spans"], s64["matches"] = setup["spans"], setup["matches"]
    return s64


def run_g1b_fd(cfg, setup, s64):
    """The FD comparison itself, on one decoded clip per pair (first pair only for speed)."""
    from stage2_sae.train_sae_umt_fused import build_loader
    from stage3_analysis.umt_vision_text_bridge import t1_clips
    from umt_fusion_grad import _fd_check
    a, b = cfg["pairs"][0]
    sp, matches = s64["spans"][f"{a}_{b}"], s64["matches"]
    loader = build_loader(t1_clips({**cfg, "max_clips": 1, "pairs": [(a, b)]}), cfg, shuffle=False)
    pv, labels = next(iter(loader))
    x9 = tower_to_layer9(setup["model"], pv.to(cfg["device"])).double().requires_grad_(True)
    ti = 0 if labels[0].item() == a else 1

    def f(x, key):
        t, _ = targets(finish_tower(s64["model"], x)[0], s64, sp, matches, ti)   # (2352, 768)
        return t[key]
    rows = {}
    for key in ("T_vtm", "T_match"):
        r = _fd_check(lambda x: f(x, key), x9, n_dir=2)
        rows[key] = max(x[2] for x in r)
    print("G1b Check B (fp64 FD):", {k: f"{v:.2e}" for k, v in rows.items()})
    return rows


def run_ap(cfg, setup):
    """Stage B: AP for every pair clip (all 3 targets) + the pair's cross-attention maps.
    Writes per_clip_ap.parquet (one row per clip: baseline targets, top features by |AP|)
    and sparse_ap/<cid>.npz (token maps + z sums for the top features, maps L9/L10)."""
    from stage2_sae.train_sae_umt_fused import build_loader
    from stage3_analysis.umt_vision_text_bridge import t1_clips
    out = cfg["out_dir"]
    (out / "sparse_ap").mkdir(parents=True, exist_ok=True)
    clips = t1_clips(cfg)
    pair_of = {c: p for p in cfg["pairs"] for c in p}
    spans, matches = setup["spans"], setup["matches"]
    rows, t0, idx = [], time.time(), 0
    for pv, labels in build_loader(clips, cfg, shuffle=False):
        x9 = tower_to_layer9(setup["model"], pv.to(cfg["device"]))
        for i, cls in enumerate(labels.tolist()):
            cid = clips[idx][0]
            idx += 1
            a, b = pair_of[cls]
            sp, ti = spans[f"{a}_{b}"], (0 if cls == a else 1)
            ap_feat, ap_tok, zsum, t_base = clip_ap(x9[i:i + 1], setup, sp, matches, ti)
            with torch.no_grad():
                vf = finish_tower(setup["model"], x9[i:i + 1].clone())[0]
            _, maps, _, _, _ = pair_pass_ref(vf, setup, a, b)
            row = dict(clip_id=cid, a=a, b=b, true_class=cls,
                       **{f"base_{k}": v for k, v in t_base.items()})
            save = {"zsum": zsum}                       # per-feature activation mass (selection)
            for k in ap_feat:
                top = np.argsort(-np.abs(ap_feat[k].astype(np.float32)))[: cfg["top_n_ap"]]
                row[f"top_{k}"] = ",".join(map(str, top))   # diagnostic only; selection is per pair
                save[f"tokap_{k}"] = ap_tok[k][:, top].astype(np.float16)
                save[f"ap_{k}"] = ap_feat[k]
            for L in (9, 10):
                save[f"map{L}"] = maps[L][ti].half().cpu().numpy()
            np.savez_compressed(out / "sparse_ap" / f"{cid}.npz", **save)
            rows.append(row)
    pd.DataFrame(rows).to_parquet(out / "per_clip_ap.parquet", index=False)
    print(f"AP stage: {len(rows)} clips in {time.time() - t0:.0f}s")


def pair_split(D, is_a):
    """Equal-weight A/B halves of A-minus-B oriented per-clip contributions D (n, C) ->
    (mean_a, mean_b, evidence, bias); the same split as umt_vision_pair_aggregate.split."""
    m_a, m_b = D[is_a].mean(0), D[~is_a].mean(0)
    return m_a, m_b, (m_a - m_b) / 2, (m_a + m_b) / 2


def discriminative(D, is_a, n_sel, max_other):
    """Per-pair verb-evidence features: heavy on one class, and on the other class silent
    (same sign, at most max_other of the heavy side) or sign-reversed; helps the true class
    (evidence > 0). Ranked by evidence. Returns (selected ids, per-feature table)."""
    m_a, m_b, ev, bias = pair_split(D, is_a)
    a_heavy = np.abs(m_a) >= np.abs(m_b)
    heavy, other = np.where(a_heavy, m_a, m_b), np.where(a_heavy, m_b, m_a)
    ratio = other / np.where(heavy == 0, 1.0, heavy)   # <0 reversed, ~0 silent, 1 same on both
    passed = (ratio <= max_other) & (ev > 0)
    order = np.argsort(-ev)
    table = pd.DataFrame(dict(feature=np.arange(len(ev)), evidence=ev, bias=bias,
                              mean_a=m_a, mean_b=m_b, ratio=ratio, passed=passed))
    return order[passed[order]][:n_sel], table


def mass_controls(mass, sel, ratio, max_other):
    """One control per selected feature: the active, non-discriminative feature (contributes
    alike on both classes: same sign, ratio > max_other — scale-free, so heavy generic
    features qualify) with the nearest mean activation mass, without replacement, heaviest
    selected first. Returns (controls, relative mass gaps)."""
    pool = set(np.where((mass > 0) & (ratio > max_other))[0].tolist()) - set(sel.tolist())
    ctrl = {}
    for f in sorted(sel, key=lambda j: -mass[j]):
        ctrl[f] = min(pool, key=lambda j: abs(mass[j] - mass[f]))
        pool.discard(ctrl[f])
    c = np.array([ctrl[f] for f in sel], dtype=int)
    return c, np.abs(mass[c] - mass[sel]) / (mass[sel] + 1e-8)


def check_selection_vs_vtc(cfg, pairs=((45, 43), (42, 40), (37, 36))):
    """Validation on option (a): run pair_split / discriminative / mass_controls on the
    on-file VTC contrast DFA (already A-minus-B oriented). evidence must reproduce
    feature_table.csv; prints the selection, controls and mass gaps per pair.
    Returns {(a, b): selected feature ids}."""
    picked = {}
    for a, b in pairs:
        d = ROOT / "outputs/analysis" / ("umt_vision_pair_contrast" + ("" if (a, b) == (45, 43) else f"_{a}_{b}"))
        need = [d / f for f in ("clip_feature_arrays_R.npz", "feature_table.csv", "per_clip.parquet")]
        if not all(p.exists() for p in need):
            print(f"VTC cross-check: skipping {a}_{b} (no contrast tables in {d.name})")
            continue                     # a cross-check only: never block the run
        R, ft = np.load(d / "clip_feature_arrays_R.npz"), pd.read_csv(d / "feature_table.csv")
        lab = pd.read_parquet(d / "per_clip.parquet").drop_duplicates("clip_id").set_index("clip_id")
        is_a = (lab.loc[[str(c) for c in R["clip_id"]], "true_class"] == a).to_numpy()
        sel, tab = discriminative(R["dfa"], is_a, cfg["n_sel"], cfg["max_other"])
        ctrl, gap = mass_controls(R["zsum"].mean(0), sel, tab.ratio.to_numpy(), cfg["max_other"])
        err = np.abs(tab.evidence.to_numpy() - ft.sort_values("feature").evidence.to_numpy()).max()
        print(f"\n{a}_{b}: n_a={is_a.sum()} n_b={(~is_a).sum()}  max|evidence - feature_table| = {err:.1e}")
        print(tab.loc[sel, ["feature", "evidence", "mean_a", "mean_b", "ratio"]]
              .assign(control=ctrl, mass_gap=gap).round(3).to_string(index=False))
        picked[(a, b)] = set(sel.tolist())
    return picked


def vtc_overlap(sel, cfg):
    """Cross-check (option a): overlap of the VTM-AP selection with the VTC-DFA selection on
    the pairs that have on-file contrast tables."""
    vtc = check_selection_vs_vtc(cfg)
    for (a, b), want in vtc.items():
        for t, g in sel[(sel.a == a) & (sel.b == b)].groupby("target"):
            both = sorted(want & set(g.feature))
            print(f"VTC vs {t} selection, {a}_{b}: {len(both)}/{len(want)} shared {both}")


def sample_g34_clips(df, n_total, pairs):
    """Stratified subset of ~n_total clips across pairs (min n_total // len(pairs) per pair)
    for the G3/G4 intervention pass. Deterministic ordering by clip_id."""
    per_pair = max(1, n_total // len(pairs))
    out = []
    for a, b in pairs:
        sub = df[(df.a == a) & (df.b == b)].sort_values("clip_id")
        out.append(sub.head(per_pair))
    res = pd.concat(out, ignore_index=True)
    if len(res) < n_total:
        rem = df[~df.clip_id.isin(res.clip_id)].sort_values("clip_id")
        res = pd.concat([res, rem.head(n_total - len(res))], ignore_index=True)
    return res


def select_pair(df, a, k_tgt, cfg):
    """One (pair, target): per-clip contribution grad.z = -AP/2 (AP is the predicted effect of
    the FLIP, -2z, so its sign is opposite to the contribution that VTC DFA measures), oriented
    A-minus-B (T is true - partner, so flip B clips); pick discriminative features + controls."""
    ap = np.stack([load_ap(cfg["out_dir"], c)[f"ap_{k_tgt}"] for c in df.clip_id])
    mass = np.stack([load_zsum(cfg["out_dir"], c) for c in df.clip_id]).mean(0)
    is_a = (df.true_class == a).to_numpy()
    contrib = -0.5 * ap * np.where(is_a, 1.0, -1.0)[:, None]
    sel, tab = discriminative(contrib, is_a, cfg["n_sel"], cfg["max_other"])
    ctrl, gap = mass_controls(mass, sel, tab.ratio.to_numpy(), cfg["max_other"])
    return tab.loc[sel].assign(target=k_tgt, rank=np.arange(len(sel)), control=ctrl,
                               mass=mass[sel], control_mass=mass[ctrl], mass_gap=gap,
                               n_a=int(is_a.sum()), n_b=int((~is_a).sum()))


def run_select(cfg):
    """Stage C (CPU): fix the intervention subset, then select per pair on the REMAINING
    clips (held out from intervention, so selection and test don't share clips).
    Writes g34_clips.csv and pair_selection.csv."""
    out = cfg["out_dir"]
    df = pd.read_parquet(out / "per_clip_ap.parquet")
    g34 = sample_g34_clips(df, cfg["g34_clips"], cfg["pairs"])
    g34.to_csv(out / "g34_clips.csv", index=False)
    rest, rows = df[~df.clip_id.isin(g34.clip_id)], []
    for a, b in cfg["pairs"]:
        sub = rest[(rest.a == a) & (rest.b == b)]
        rows += [select_pair(sub, a, k, cfg).assign(a=a, b=b) for k in ("T_vtm", "T_match")]
    sel = pd.concat(rows, ignore_index=True)
    sel.to_csv(out / "pair_selection.csv", index=False)
    print(sel.groupby(["a", "b", "target"]).agg(n=("feature", "size"), top=("feature", "first"),
          max_gap=("mass_gap", "max")).to_string())
    vtc_overlap(sel, cfg)


def apply_edit(x9, z, D, feats, dose=2.0):
    """Raw-stream edit x' = x9 - dose * sum(z_f * d_f) over the selected features.
    Vectorized: z[:, feats] @ D[feats] over the active token sequence."""
    if len(feats) == 0:
        return x9
    idx = torch.as_tensor(feats, dtype=torch.long, device=z.device)
    delta = dose * (z[:, idx] @ D[idx]).unsqueeze(0)
    return x9 - delta


def eval_intervention(ctx, feats, dose, route=None):
    """One intervention forward (full, or route-split into one projection), no grad.
    ctx holds the clip's x9, z, D, setup, sp, matches, ti, t_base, vis_base.
    Returns ({delta_<target>: real change} for every target, edited vision tokens) — the
    tokens let callers read attention maps without another tower pass."""
    s = ctx["setup"]
    with torch.no_grad():
        x_int = apply_edit(ctx["x9"], ctx["z"], ctx["D"], feats, dose=dose)
        vis_int = finish_tower(s["model"], x_int)[0]
        vis, alt = (ctx["vis_base"], vis_int) if route is not None else (vis_int, None)
        t_int, _ = targets(vis, s, ctx["sp"], ctx["matches"], ctx["ti"], route=route, alt_vis=alt)
    d = {f"delta_{k}": float(t_int[k] - ctx["t_base"][k]) for k in ctx["t_base"]}
    return d, vis_int


def att_mass(maps, mask, ti, prefix):
    """Cross-attention mass (head-mean, query-averaged over real text tokens; Phase 1's
    pair_pass maps) that the true and partner templates put on the tokens in mask, per
    fusion layer 9-11. Each map sums to 1 over the 2352 tokens."""
    return {f"{prefix}_{nm}_L{L}": float(maps[L][r][mask].sum())
            for L in (9, 10, 11) for nm, r in (("true", ti), ("partner", 1 - ti))}


def ids(s):
    """'3,17,42' -> [3, 17, 42] (parquet stores feature lists as comma strings)."""
    return [int(f) for f in str(s).split(",") if f]


def run_clip_interventions(x9, cid, cls, picks, setup, sp, matches, D, cfg):
    """Singles, top-k sets vs their mass-matched controls, and route splits, one clip.
    picks = {target: (selected, controls)} — the clip's pair's fixed lists, rank-aligned."""
    ti = 0 if cls == sp["a"] else 1
    with torch.no_grad():
        vis_base = finish_tower(setup["model"], x9)[0]
        t_base, _ = targets(vis_base, setup, sp, matches, ti)
        _, z = setup["sae"].encode((x9[0] - setup["dim_mean"]).float())
    ctx = dict(x9=x9, z=z, D=D, setup=setup, sp=sp, matches=matches, ti=ti,
               t_base={k: float(v) for k, v in t_base.items()}, vis_base=vis_base,
               maps_base=pair_pass_ref(vis_base, setup, sp["a"], sp["b"])[1])
    recs = []
    for k_tgt, (top, ctrl) in picks.items():
        tag = dict(clip_id=cid, a=sp["a"], b=sp["b"], true_class=cls, target_src=k_tgt)
        recs += [{**tag, **r} for r in eval_singles(ctx, top, ctrl)]
        recs += [{**tag, **r} for r in eval_sets(ctx, top, ctrl, cfg)]
    return recs


def eval_singles(ctx, top, ctrl):
    """G3 pool: each selected feature and each mass-matched control alone, at both doses.
    A feature inactive on this clip gives a zero edit (and delta 0) — expected for gated ones.
    Also records the attention the templates put on the feature's active tokens (tok_frac =
    share of tokens active; att_base_* at baseline, att_int_* under this edit). At fusion L9
    the text queries don't depend on the video, so the L9 change is purely the key route."""
    s, sp, ti, recs = ctx["setup"], ctx["sp"], ctx["ti"], []
    for kind, fs in (("single", top), ("control_single", ctrl)):
        for f in fs:
            mask = ctx["z"][:, f] > 0
            base = dict(tok_frac=float(mask.float().mean()), **att_mass(ctx["maps_base"], mask, ti, "att_base"))
            for dose in (1.0, 2.0):
                d, vis_int = eval_intervention(ctx, [f], dose)
                with torch.no_grad():
                    maps_int = pair_pass_ref(vis_int, s, sp["a"], sp["b"])[1]
                recs.append(dict(kind=kind, feats=str(f), k=1, dose=dose, route="full", **d,
                                 **base, **att_mass(maps_int, mask, ti, "att_int")))
    return recs


def eval_sets(ctx, top, ctrl, cfg):
    """G4: top[:k] vs its k matched controls ctrl[:k] at both doses, plus the full-flip
    route splits (key-only / value-only into fusion L9, L10, or all of 9-11) for top[:k]."""
    recs = []
    for ks in cfg["set_sizes"]:
        if ks > len(top):
            continue
        for kind, fs in (("set", top[:ks]), ("control_set", ctrl[:ks])):
            for dose in (1.0, 2.0):
                recs.append(dict(kind=kind, feats=",".join(map(str, fs)), k=ks, dose=dose,
                                 route="full", **eval_intervention(ctx, fs, dose)[0]))
        for L in cfg["route_layers"]:
            for proj in ("key", "value"):
                recs.append(dict(kind="route_split", feats=",".join(map(str, top[:ks])), k=ks,
                                 dose=2.0, route=f"L{L}_{proj}",
                                 **eval_intervention(ctx, top[:ks], 2.0, route=(proj, L))[0]))
    return recs


def run_intervene(cfg, setup):
    """Stage D: G3 (agreement across doses) and G4 (causal vs control, key/val route split)
    on the held-out subset fixed by run_select, with each pair's selected features and
    controls. Writes per_clip_interventions.parquet."""
    from stage2_sae.train_sae_umt_fused import build_loader
    sampled = pd.read_csv(cfg["out_dir"] / "g34_clips.csv", dtype={"clip_id": str})
    sel = pd.read_csv(cfg["out_dir"] / "pair_selection.csv").sort_values("rank")
    picks = {(a, b): {t: (g.feature.tolist(), g.control.tolist()) for t, g in s.groupby("target")}
             for (a, b), s in sel.groupby(["a", "b"])}
    pair_of = {c: p for p in cfg["pairs"] for c in p}
    spans, matches = setup["spans"], setup["matches"]
    D = setup["sae"].get_dictionary()
    rows_clips = [(r.clip_id, int(r.true_class)) for _, r in sampled.iterrows()]
    loader = build_loader(rows_clips, cfg, shuffle=False)
    records, t0, idx = [], time.time(), 0
    for pv, _ in loader:
        x9 = tower_to_layer9(setup["model"], pv.to(cfg["device"]))
        for i in range(len(pv)):
            row = sampled.iloc[idx]; idx += 1
            cid, cls = row.clip_id, int(row.true_class)
            a, b = pair_of[cls]
            records += run_clip_interventions(x9[i:i + 1], cid, cls, picks[(a, b)], setup,
                                              spans[f"{a}_{b}"], matches, D, cfg)
    pd.DataFrame(records).to_parquet(cfg["out_dir"] / "per_clip_interventions.parquet", index=False)
    print(f"Interventions: {len(sampled)} clips ({len(records)} rows) in {time.time() - t0:.0f}s")


@lru_cache(maxsize=None)
def load_ap(out_dir, cid):
    """{ap_<target>: (6144,) fp32} full-flip AP vectors for one clip, loaded once."""
    with np.load(out_dir / "sparse_ap" / f"{cid}.npz") as z:
        return {k: z[k].astype(np.float32) for k in z.files if k.startswith("ap_")}


@lru_cache(maxsize=None)
def load_zsum(out_dir, cid):
    """(6144,) per-feature activation mass (z summed over tokens) for one clip."""
    with np.load(out_dir / "sparse_ap" / f"{cid}.npz") as z:
        return z["zsum"].astype(np.float32)


def ap_lookup(out_dir, cid, key, feats):
    """Sum of the stored full-flip AP vector `key` (e.g. ap_T_vtm) over feats for one clip."""
    return float(load_ap(out_dir, cid)[key][feats].sum())


def agg_g3_agreement(df_int, out_dir):
    """G3: Spearman and slope of real delta vs AP-predicted (AP * dose/2), per target and dose.
    pool 'all' = top-N + random mass-matched controls (the brief's 'plus a random sample');
    'top' = top-N only, kept for reference (narrower AP range, so lower rank correlation)."""
    from scipy.stats import spearmanr
    singles = df_int[df_int.kind.isin(["single", "control_single"])].copy()
    singles["pred"] = [ap_lookup(out_dir, r.clip_id, f"ap_{r.target_src}", [int(r.feats)]) * r.dose / 2
                       for r in singles.itertuples()]
    rows = []
    for (tgt, dose), g in singles.groupby(["target_src", "dose"]):
        for pool, sub in (("all", g), ("top", g[g.kind == "single"])):
            ok = len(sub) > 2
            rows.append(dict(target=tgt, dose=dose, pool=pool, n=len(sub),
                             spearman=float(spearmanr(sub.pred, sub.d_self).statistic) if ok else np.nan,
                             slope=float(np.polyfit(sub.pred, sub.d_self, 1)[0]) if ok else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "g3_agreement.csv", index=False)
    print("\n== G3 Agreement (AP vs Real) ==\n", res.round(3).to_string(index=False))
    return res


def agg_g4_causal_and_routes(df_int, out_dir):
    """G4: full-flip effect of top[:k] vs its own size-k mass-matched control (sets only, so
    k=1 isn't double-counted with the G3 singles), then route shares per clip and k."""
    full = df_int[df_int.kind.isin(["set", "control_set"]) & (df_int.dose == 2.0)]
    rows_c = []
    for (tgt, k), g in full.groupby(["target_src", "k"]):
        t, c = g[g.kind == "set"].d_self, g[g.kind == "control_set"].d_self
        rows_c.append(dict(target=tgt, k=k, n=len(t), mean_abs_target=t.abs().mean(),
                           mean_abs_control=c.abs().mean(), mean_signed_target=t.mean(),
                           mean_signed_control=c.mean(),
                           ratio=t.abs().mean() / (c.abs().mean() + 1e-6)))
    res_c = pd.DataFrame(rows_c)
    res_c.to_csv(out_dir / "g4_causal.csv", index=False)
    print("\n== G4 Causal Effect Beyond Control ==\n", res_c.round(3).to_string(index=False))

    # Route shares: |route delta| / |full delta|, joined on the unique (clip, target, k) set row
    key = ["clip_id", "target_src", "k"]
    sets = full[full.kind == "set"][key + ["d_self"]].rename(columns={"d_self": "d_full"})
    routes = df_int[df_int.kind == "route_split"].merge(sets, on=key, how="inner")
    routes["share"] = routes.d_self.abs() / (routes.d_full.abs() + 1e-6)
    res_r = (routes.groupby(["target_src", "route", "k"]).share
             .agg(["mean", "median", "count"]).reset_index())
    res_r.to_csv(out_dir / "g4_routes.csv", index=False)
    print("\n== G4 Route Shares (Key vs Value) ==\n", res_r.round(3).to_string(index=False))
    return res_c, res_r


def agg_g2_completeness(df_int, out_dir):
    """G2 (report only): unexplained fraction |delta_real - sum AP| / |delta_real| for the top
    sets, per measured target = per layer (T_match9 = L9, T_match = L10, T_vtm), both doses.
    AP is taken at raw x9 and the edit only moves along feature directions, so the SAE error
    term is unchanged by the edit and contributes zero to delta: it is held, not dropped."""
    sets = df_int[df_int.kind == "set"]
    rows = []
    for col in ("T_vtm", "T_match", "T_match9"):
        pred = np.array([ap_lookup(out_dir, r.clip_id, f"ap_{col}", ids(r.feats)) * r.dose / 2
                         for r in sets.itertuples()])
        real = sets[f"delta_{col}"].to_numpy()
        g = sets.assign(resid=np.abs(real - pred) / (np.abs(real) + 1e-6))
        rows.append(g.groupby(["target_src", "k", "dose"]).resid.agg(["mean", "median"])
                    .reset_index().assign(measured=col))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out_dir / "g2_completeness.csv", index=False)
    print("\n== G2 Completeness Residual ==\n", res.round(3).to_string(index=False))
    return res, g2_route_split(df_int, out_dir)


def agg_attention(df_int, out_dir):
    """Text-side attention on each feature's own tokens (singles, flip dose, clips where the
    feature is active), per pair, target, feature and true class: tok_frac, baseline mass per
    template/layer, enrichment = mass / tok_frac (1 = uniform attention), and the change
    under the flip (datt; at L9 purely the key route). Writes attention_on_features.csv."""
    s = df_int[df_int.kind.isin(["single", "control_single"]) & (df_int.dose == 2.0)
               & (df_int.tok_frac > 0)].copy()
    cols = ["tok_frac"]
    for L in (9, 10, 11):
        for nm in ("true", "partner"):
            base = s[f"att_base_{nm}_L{L}"]
            s[f"enrich_{nm}_L{L}"] = base / s.tok_frac
            s[f"datt_{nm}_L{L}"] = s[f"att_int_{nm}_L{L}"] - base
            cols += [f"att_base_{nm}_L{L}", f"enrich_{nm}_L{L}", f"datt_{nm}_L{L}"]
    g = s.groupby(["a", "b", "target_src", "kind", "feats", "true_class"])
    res = g[cols].mean().assign(n_active=g.size()).reset_index()
    res.to_csv(out_dir / "attention_on_features.csv", index=False)
    show = ["a", "b", "target_src", "kind", "feats", "true_class", "n_active", "tok_frac",
            "enrich_true_L9", "datt_true_L9", "enrich_true_L10", "datt_true_L10"]
    print("\n== Attention on feature tokens ==\n", res[show].round(3).to_string(index=False))
    return res


def agg_additivity(df_int, out_dir):
    """Overlap between features: real set delta vs the sum of its members' real single deltas
    (same clip, target, dose). ratio < 1 = redundant (members carry overlapping signal),
    > 1 = synergistic. Selected vs control sets. Sanity: k = 1 must give ratio exactly 1."""
    singles = df_int[df_int.kind.isin(["single", "control_single"])]
    lut = singles.set_index(["clip_id", "target_src", "dose", "feats"]).d_self.to_dict()
    sets = df_int[df_int.kind.isin(["set", "control_set"])].copy()
    sets["sum_singles"] = [sum(lut[(r.clip_id, r.target_src, r.dose, str(f))] for f in ids(r.feats))
                           for r in sets.itertuples()]
    sets["nonadd"] = (sets.d_self - sets.sum_singles).abs() / (sets.d_self.abs() + 1e-6)
    res = (sets.groupby(["target_src", "kind", "k", "dose"])
           .agg(n=("d_self", "size"), mean_set=("d_self", "mean"),
                mean_sum_singles=("sum_singles", "mean"), median_nonadd=("nonadd", "median"))
           .reset_index())
    res["ratio"] = res.mean_set / res.mean_sum_singles.where(res.mean_sum_singles.abs() > 1e-9)
    res.to_csv(out_dir / "g2_additivity.csv", index=False)
    print("\n== Set vs sum of singles (overlap) ==\n", res.round(3).to_string(index=False))
    return res


def g2_route_split(df_int, out_dir):
    """G2 split: signed fraction of each full-flip set effect carried by the key-only and the
    value-only patch of all fusion layers; nonadditive = 1 - key - value (key x value
    interaction + nonlinearity). Medians: fractions blow up when the full delta is ~0."""
    key = ["clip_id", "target_src", "k"]
    full = df_int[(df_int.kind == "set") & (df_int.dose == 2.0)][key + ["d_self"]]
    r = (df_int[df_int.route.isin(["Lall_key", "Lall_value"])]
         .pivot_table(index=key, columns="route", values="d_self").reset_index().merge(full, on=key))
    r["key_frac"], r["value_frac"] = r.Lall_key / r.d_self, r.Lall_value / r.d_self
    r["nonadditive"] = 1 - r.key_frac - r.value_frac
    res = r.groupby(["target_src", "k"])[["key_frac", "value_frac", "nonadditive"]].median().reset_index()
    res.to_csv(out_dir / "g2_route_split.csv", index=False)
    print("\n== G2 Key / Value / Non-additive split ==\n", res.round(3).to_string(index=False))
    return res


def run_aggregate(cfg):
    """Stage E: aggregate G2 completeness + additivity, G3 agreement, G4 causal / route
    tables, and attention on feature tokens.
    d_self = each row's delta on the target its features were selected by."""
    out = cfg["out_dir"]
    df_int = pd.read_parquet(out / "per_clip_interventions.parquet")
    df_int["d_self"] = np.where(df_int.target_src == "T_vtm", df_int.delta_T_vtm, df_int.delta_T_match)
    agg_g3_agreement(df_int, out)
    agg_g4_causal_and_routes(df_int, out)
    agg_g2_completeness(df_int, out)
    agg_additivity(df_int, out)
    agg_attention(df_int, out)


def main():
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    stage = cfg["stage"]
    print(f"Phase 3: stage={stage} device={cfg['device']} clips_cap={cfg['max_clips']}")
    if stage in ("g1b", "all"):
        setup = phase3_setup(cfg)
        run_g1b(cfg, setup)
        s64 = g1b_check_b(cfg, setup)
        run_g1b_fd(cfg, setup, s64)
    if stage in ("ap", "all"):
        setup = phase3_setup(cfg) if stage != "all" else setup
        run_ap(cfg, setup)
    if stage == "selcheck":                      # CPU validation of selection on on-file VTC DFA
        check_selection_vs_vtc(cfg)
    if stage in ("select", "all"):
        run_select(cfg)
    if stage in ("intervene", "all"):
        setup = phase3_setup(cfg) if stage != "all" else setup
        run_intervene(cfg, setup)
    if stage in ("aggregate", "all"):
        run_aggregate(cfg)


if __name__ == "__main__":
    main()
