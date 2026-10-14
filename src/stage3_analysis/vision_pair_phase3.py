"""Phase 3 ported to the classifier backbones (VideoMAE, TimeSformer; UMT's VTC head too):
which layer-L SAE features does the CLASSIFIER use to tell a pair apart, and how? Same method
as umt_bridge_phase3.py with the text side replaced by T_cls = logit(true) - logit(partner).
Ports: AP at the raw layer-L residual, per-pair discriminative selection + mass-matched
controls, raw-stream edits, set additivity, location by class. Not ported (no text side):
VTM, T_match, route split, cross-attention measures.
--all-pairs runs the 8 pairs in one job (model loaded once, ~3.6k clips, < 30 min) and then
`combine` stacks them; --pair / --pair-index runs one pair.
Usage: python src/stage3_analysis/vision_pair_phase3.py --model videomae --layer 9 --all-pairs
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

CFG = {   # env vars are the defaults for the command-line options (sbatch --export works too)
    "model":     os.environ.get("MODEL", "videomae"),    # videomae | timesformer | umt (VTC head)
    "layer":     int(os.environ.get("LAYER", 9)),        # any layer with a k64 SAE
    "stage":     os.environ.get("STAGE", "all"),         # all | ap+select+intervene+aggregate | combine
    "max_clips": int(os.environ.get("MAX_CLIPS", 0)),    # smoke cap (writes to <run>_smoke)
    "g34_clips": int(os.environ.get("G34_CLIPS", 37)),   # intervention clips per pair (UMT P3: 300/8)
    "n_sel": 10, "max_other": 1 / 3, "set_sizes": (1, 5, 10),   # as umt_bridge_phase3
    "edit_batch": 16,                                    # edits per batched forward
    "pairs": [(6, 171), (46, 5), (14, 172), (168, 165), (45, 43), (44, 41), (42, 40), (37, 36)],
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "device": "cuda" if torch.cuda.is_available() else "cpu", "num_workers": 4,
}


def parse_args(cfg):
    """--model / --layer / --pair A B (or --pair-index i into CFG['pairs'], default the SLURM
    array task id) / --stage / --max-clips. Output: vision_pair_phase3/<model>_L<layer>/<a>_<b>/."""
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=cfg["model"], choices=["videomae", "timesformer", "umt"])
    p.add_argument("--layer", type=int, default=cfg["layer"])
    p.add_argument("--pair", type=int, nargs=2, default=[45, 43])
    p.add_argument("--pair-index", type=int, default=os.environ.get("PAIR_INDEX"))
    p.add_argument("--all-pairs", action="store_true", help="all 8 pairs in one run, then combine")
    p.add_argument("--stage", default=cfg["stage"])
    p.add_argument("--max-clips", type=int, default=cfg["max_clips"])
    a = p.parse_args()
    pair = cfg["pairs"][a.pair_index] if a.pair_index is not None else tuple(a.pair)
    cfg.update(model=a.model, layer=a.layer, stage=a.stage, max_clips=a.max_clips,
               pair_list=list(cfg["pairs"]) if a.all_pairs else [pair])
    cfg["run_dir"] = ROOT / "outputs/analysis/vision_pair_phase3" / (
        f"{a.model}_L{a.layer}" + ("_smoke" if a.max_clips else ""))
    return set_pair(cfg, cfg["pair_list"][0])


def set_pair(cfg, pair):
    """Point cfg at one pair: cfg['pair'] and its output dir <run_dir>/<a>_<b>/."""
    cfg["pair"], cfg["out_dir"] = tuple(pair), cfg["run_dir"] / f"{pair[0]}_{pair[1]}"
    return cfg


def setup(cfg):
    """Model + processor (DFAEngine's registry path), the layer-L SAE at its trained threshold,
    dim_mean, decoder D, and one forward hook on layer L (registry layer_getter: layer index
    only, no model internals). st['new'] set -> the hook splices those patch tokens in."""
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    m, dev = MODEL_REGISTRY[cfg["model"]], cfg["device"]
    ckpt = CHECKPOINT_REGISTRY[(cfg["model"], "ssv2")]
    model = m["model_class"].from_pretrained(ckpt).to(dev).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    res = resolve_sae_checkpoint(cfg["model"], cfg["layer"], dataset_name="ssv2", sae_k=64, job_label="7ep")
    sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], dev, sae_k=64)
    st = dict(model=model, processor=m["processor_class"].from_pretrained(ckpt), sae=sae,
              dm=torch.load(res["dim_mean_path"], weights_only=True).to(dev).float(),
              D=sae.get_dictionary(), new=None, x=None)
    m["layer_getter"](model, cfg["layer"]).register_forward_hook(make_hook(st, m["cls_offset"]))
    print(f"SAE {Path(res['sae_path']).name}  threshold {float(sae.running_threshold):.3f}", flush=True)
    return st


def make_hook(st, off):
    """Layer-L hook: st['x'] <- patch tokens (B, N, D). If st['new'] (B, N, D) is set, return it
    in place of the patch tokens; CLS rows (TF: off = 1) are kept. Tuple or tensor outputs."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        st["x"] = h[:, off:]
        if st["new"] is None:
            return None
        new = st["new"].to(h.dtype)
        if off:
            new = torch.cat([h[:, :off], new], dim=1)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
    return hook


def forward_T(st, pv, new, t, p):
    """(B,) logit contrast t - p, with `new` (B, N, D) spliced at layer L (pv repeated to B
    rows so every layer sees the same batch); new=None -> plain forward (B = 1)."""
    B = 1 if new is None else new.shape[0]
    st["new"] = new
    try:
        logits = st["model"](pixel_values=pv.expand(B, *pv.shape[1:])).logits
    finally:
        st["new"] = None
    return logits[:, t] - logits[:, p]


def encode_clip(st, pv, t, p):
    """Plain forward: baseline T, layer-L patch tokens x (N, D) fp32, SAE codes z0 (N, C)."""
    with torch.no_grad():
        T = float(forward_T(st, pv, None, t, p)[0])
        x = st["x"][0].float()
        z0 = st["sae"].encode(x - st["dm"])[1]
    return T, x, z0


def clip_ap(st, pv, t, p):
    """AP at the RAW layer-L residual: x' = x + decode(z - z0) (= x at z = z0, SAE error held),
    T = logit(t) - logit(p); AP = dT/dz * (-2 z) summed over tokens = predicted effect of the
    flip (contribution = -AP/2). Returns (ap (C,), zsum (C,), T_base)."""
    T, x, z0 = encode_clip(st, pv, t, p)
    z = z0.clone().requires_grad_(True)
    T_g = forward_T(st, pv, (x + st["sae"].decode(z - z0)).unsqueeze(0), t, p)[0]
    g = torch.autograd.grad(T_g, z)[0]
    return (g * (-2.0 * z0)).sum(0).cpu().numpy(), z0.sum(0).cpu().numpy(), T


def run_ap(cfg, st):
    """Stage 1: AP for every val clip of the pair (R condition, each model's own sampler and
    processor via umt_vision_pair_contrast's loaders). sparse_ap/<cid>.npz holds ap_T_cls and
    zsum — the keys umt_bridge_phase3's loaders read; per_clip_ap.parquet the baselines."""
    from umt_vision_pair_contrast import clip_loader, pair_clips
    a, b = cfg["pair"]
    rows, out, recs, t0 = pair_clips(cfg), cfg["out_dir"], [], time.time()
    (out / "sparse_ap").mkdir(parents=True, exist_ok=True)
    for (cid, cls), (pv, _) in zip(rows, clip_loader(rows, cfg, "R", st["processor"])):
        t, p = (a, b) if cls == a else (b, a)
        ap, zsum, T = clip_ap(st, pv.to(cfg["device"]), t, p)
        np.savez_compressed(out / "sparse_ap" / f"{cid}.npz", ap_T_cls=ap, zsum=zsum)
        recs.append(dict(clip_id=cid, a=a, b=b, true_class=cls, base_T_cls=T))
    pd.DataFrame(recs).to_parquet(out / "per_clip_ap.parquet", index=False)
    print(f"AP: {len(recs)} clips in {time.time() - t0:.0f}s", flush=True)


def run_select(cfg):
    """Stage 2 (CPU): intervention subset = the pair's first g34_clips by clip id (UMT Phase 3's
    rule, so clips are largely shared across backbones; capped at 1/3 for smoke runs); select on
    the rest with umt_bridge_phase3's discriminative + mass_controls, contribution = -AP/2."""
    from umt_bridge_phase3 import discriminative, load_ap, load_zsum, mass_controls
    (a, b), out = cfg["pair"], cfg["out_dir"]
    df = pd.read_parquet(out / "per_clip_ap.parquet").sort_values("clip_id")
    n = min(cfg["g34_clips"], len(df) // 3)
    df.head(n).to_csv(out / "g34_clips.csv", index=False)
    rest = df.iloc[n:]
    ap = np.stack([load_ap(out, c)["ap_T_cls"] for c in rest.clip_id])
    mass = np.stack([load_zsum(out, c) for c in rest.clip_id]).mean(0)
    is_a = (rest.true_class == a).to_numpy()
    sel, tab = discriminative(-0.5 * ap * np.where(is_a, 1.0, -1.0)[:, None], is_a,
                              cfg["n_sel"], cfg["max_other"])
    ctrl, gap = mass_controls(mass, sel, tab.ratio.to_numpy(), cfg["max_other"])
    tab.loc[sel].assign(a=a, b=b, target="T_cls", rank=np.arange(len(sel)), control=ctrl, mass=mass[sel],
                        control_mass=mass[ctrl], mass_gap=gap, n_a=int(is_a.sum()), n_b=int((~is_a).sum())
                        ).to_csv(out / "pair_selection.csv", index=False)
    print(f"selected {sel.tolist()}  controls {ctrl.tolist()}  max mass gap {gap.max():.2f}", flush=True)


def edit_list(sel, ctrl, cfg):
    """[(kind, feats, k, dose)]: each selected feature and control alone (G3 pool), and top[:k]
    vs ctrl[:k] sets (G4, additivity), all at dose 1 (ablate) and 2 (flip)."""
    out = [(kind, [f], 1, d) for kind, fs in (("single", sel), ("control_single", ctrl))
           for f in fs for d in (1.0, 2.0)]
    for k in cfg["set_sizes"]:
        if k <= len(sel):
            out += [(kind, fs[:k], k, d) for kind, fs in (("set", sel), ("control_set", ctrl))
                    for d in (1.0, 2.0)]
    return out


def eval_edits(st, pv, x, z0, edits, t, p, T_base, bs):
    """Real change in T per edit: raw-stream x' = x - dose * z[:, F] @ D[F] spliced at layer L,
    bs edits per batched forward; no grad."""
    deltas = []
    with torch.no_grad():
        for i in range(0, len(edits), bs):
            new = torch.stack([x - d * (z0[:, fs] @ st["D"][fs]) for _, fs, _, d in edits[i:i + bs]])
            deltas += (forward_T(st, pv, new, t, p) - T_base).tolist()
    return deltas


def location(z0, f, model):
    """Where feature f fires: activation mass per temporal position (fr0..: VM tubelets, TF/UMT
    frames; gather_by_position handles each token layout), activation-weighted centroid
    (cy, cx) on the 14x14 grid, token share."""
    from ToT_utils import gather_by_position
    a = gather_by_position(z0[:, f], model).float()                  # (positions, 196)
    tot, pos, grid = float(a.sum()), torch.arange(14, device=a.device, dtype=a.dtype), a.sum(0).view(14, 14)
    out = {f"fr{i}": float(v) for i, v in enumerate(a.sum(1))}
    out["cy"] = float((grid.sum(1) * pos).sum()) / tot if tot > 0 else float("nan")
    out["cx"] = float((grid.sum(0) * pos).sum()) / tot if tot > 0 else float("nan")
    out["tok_frac"] = float((z0[:, f] > 0).float().mean())
    return out


def run_clip(st, pv, cid, cls, sel, ctrl, cfg):
    """Every edit on one clip, as records in umt_bridge_phase3's column layout (target_src
    T_cls, delta_T_cls; singles carry location columns) so its aggregators apply unchanged."""
    a, b = cfg["pair"]
    t, p = (a, b) if cls == a else (b, a)
    T_base, x, z0 = encode_clip(st, pv, t, p)
    edits = edit_list(sel, ctrl, cfg)
    recs = []
    for (kind, fs, k, dose), d in zip(edits, eval_edits(st, pv, x, z0, edits, t, p, T_base, cfg["edit_batch"])):
        rec = dict(clip_id=cid, a=a, b=b, true_class=cls, target_src="T_cls", kind=kind,
                   feats=",".join(map(str, fs)), k=k, dose=dose, route="full", delta_T_cls=d)
        recs.append({**rec, **location(z0, fs[0], cfg["model"])} if kind.endswith("single") else rec)
    return recs


def run_intervene(cfg, st):
    """Stage 3: every edit on the held-out intervention clips -> per_clip_interventions.parquet."""
    from umt_vision_pair_contrast import clip_loader
    out = cfg["out_dir"]
    picks = pd.read_csv(out / "pair_selection.csv").sort_values("rank")
    sel, ctrl = picks.feature.tolist(), picks.control.tolist()
    g34 = pd.read_csv(out / "g34_clips.csv", dtype={"clip_id": str})
    rows, recs, t0 = list(zip(g34.clip_id, g34.true_class.astype(int))), [], time.time()
    for (cid, cls), (pv, _) in zip(rows, clip_loader(rows, cfg, "R", st["processor"])):
        recs += run_clip(st, pv.to(cfg["device"]), cid, cls, sel, ctrl, cfg)
    pd.DataFrame(recs).to_parquet(out / "per_clip_interventions.parquet", index=False)
    print(f"interventions: {len(rows)} clips, {len(recs)} edits in {time.time() - t0:.0f}s", flush=True)


def run_aggregate(cfg):
    """Stage 4 (CPU): umt_bridge_phase3's aggregators on the T_cls records: G3 agreement, G4
    causal vs control, set additivity, location by class (feature_location_by_class.csv)."""
    from umt_bridge_phase3 import agg_additivity, agg_g3_agreement, agg_g4_causal_and_routes, agg_location
    df = pd.read_parquet(cfg["out_dir"] / "per_clip_interventions.parquet")
    df["d_self"] = df.delta_T_cls
    for fn in (agg_g3_agreement, agg_g4_causal_and_routes, agg_additivity, agg_location):
        fn(df, cfg["out_dir"])


def run_combine(cfg):
    """Stage `combine` (CPU, after all pair tasks; runs locally on synced CSVs): stack each
    pair's CSVs into <run>/<name>_all.csv and print the cross-pair summary."""
    run, parts = cfg["run_dir"], {}
    pair_dirs = sorted(d for d in run.iterdir() if d.is_dir())
    for name in ("pair_selection", "g3_agreement", "g4_causal", "g2_additivity", "feature_location_by_class"):
        dfs = [pd.read_csv(d / f"{name}.csv").assign(pair=d.name) for d in pair_dirs if (d / f"{name}.csv").exists()]
        if dfs:
            parts[name] = pd.concat(dfs, ignore_index=True)
            parts[name].to_csv(run / f"{name}_all.csv", index=False)
    print(f"{run.name}: {len(pair_dirs)} pair dirs, tables {sorted(parts)}")
    summary(parts, run)


def summary(parts, run):
    """Per pair, the numbers compared with UMT Phase 3: share of selected features that are
    sign-reversed (UMT text side 70-75%, UMT VTC 10-40%) and gated (token ratio >= 2); among
    reversed, same-timing (r > 0.8, token ratio < 1.5) vs timing-flipped (r < 0); G4 ratio
    (k = 10, flip) and G3 Spearman (flip, pool all). Writes summary.csv."""
    sel, loc = parts["pair_selection"], parts["feature_location_by_class"]
    sl, rv = loc[loc.type != "control"], loc[loc.type == "reversed"]
    s = pd.DataFrame({
        "reversed_share": sel.assign(v=sel.ratio < 0).groupby("pair").v.mean(),
        "gated_share": sl.assign(v=sl.tok_ratio >= 2).groupby("pair").v.mean(),
        "rev_same_timing": rv.assign(v=(rv.frame_corr > 0.8) & (rv.tok_ratio < 1.5)).groupby("pair").v.mean(),
        "rev_timing_flipped": rv.assign(v=rv.frame_corr < 0).groupby("pair").v.mean(),
        "g4_ratio_k10": parts["g4_causal"].query("k == 10").set_index("pair").ratio,
        "g3_spearman_flip": parts["g3_agreement"].query("dose == 2.0 and pool == 'all'").set_index("pair").spearman})
    s.loc["all"] = s.mean()
    s.to_csv(run / "summary.csv")
    rec = sel.feature.value_counts()
    print(s.round(2).to_string(), "\nfeatures selected in >= 3 pairs:", rec[rec >= 3].to_dict())


def main():
    torch.backends.cuda.matmul.allow_tf32 = False      # match the verified fp32 paths
    torch.backends.cudnn.allow_tf32 = False
    cfg = parse_args(dict(CFG))
    stages = ({"ap", "select", "intervene", "aggregate"} if cfg["stage"] == "all"
              else set(cfg["stage"].split("+")))       # "+" not ",": sbatch --export splits on commas
    print(f"model={cfg['model']} layer={cfg['layer']} pairs={cfg['pair_list']} stages={sorted(stages)} "
          f"device={cfg['device']} run={cfg['run_dir']}", flush=True)
    if "combine" in stages:
        return run_combine(cfg)
    st = setup(cfg) if stages & {"ap", "intervene"} else None   # model + SAE loaded once
    for pair in cfg["pair_list"]:
        set_pair(cfg, pair)["out_dir"].mkdir(parents=True, exist_ok=True)
        print(f"\n=== pair {pair} ===", flush=True)
        for name, fn in (("ap", run_ap), ("select", run_select), ("intervene", run_intervene), ("aggregate", run_aggregate)):
            if name in stages:
                fn(cfg, st) if name in ("ap", "intervene") else fn(cfg)
    if len(cfg["pair_list"]) > 1:
        run_combine(cfg)


if __name__ == "__main__":
    main()
