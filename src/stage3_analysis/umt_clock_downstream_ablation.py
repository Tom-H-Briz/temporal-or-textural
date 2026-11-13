"""
Clock ablation with downstream capture (10/11/26). Ablate (spliced) at LAYER (5 or 7); every arm's forward also
records, for each later SAE layer, the per-clip x frame profile of all 6,144 latents (encode only, no second splice).
Clips: every val clip of 5 verb pairs (45/43, 168/165, 93/86, 44/41, 0/32), R + Rev; role = sel / test from the pair
contrast sets (report test-only alongside). Arms: baseline, batch_base, the selection arms, clock (ramps + near-miss).
Add-ons: (a) block LAYER+1 attention frame x frame per head, every row; (b) per-token residual norm at LAYER and each
capture layer, baseline; (c) per-token summed comb-latent activation at LAYER, baseline. SPLICE=0 (raw, own dir) adds
(d) attribution of each clip's home key features (sets json attr_targets) to every LAYER latent per frame.
Every run also stores the baseline residual per frame / tubelet (mean over space) at LAYER and each capture layer
(res_L*, the position-fraction ruler). MODEL=videomae (11/11/26): VM positions = 8 tubelets, arms = spikes / others
(vm_clock_downstream_sets.json), + pairs 8/9 (report separately), + a baseline-only C1 pass for res_L* only.
Usage: [MODEL=videomae] LAYER=5 [TWOBYTWO=1: arms clock / builders / both, use a CLASSES subset + own OUT_DIR] [SPLICE=0] [MAX_CLIPS=5] python -u src/stage3_analysis/umt_clock_downstream_ablation.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

L = int(os.environ.get("LAYER", 5))
RAW = "" if os.environ.get("SPLICE", "1") == "1" else "_raw"     # SPLICE=0: raw-stream subtraction, own dir
MODEL = os.environ.get("MODEL", "umt")                           # umt | videomae (VM run, 11/11/26)
TAG, VM = {"umt": "umt", "videomae": "vm"}[MODEL], MODEL == "videomae"
CFG = {
    "layer": L, "capture": [c for c in (7, 9) if c > L], "attn_block": L + 1,
    "model": MODEL, "sae_k": 64, "job_label": "7ep", "splice": os.environ.get("SPLICE", "1") == "1", "dose": 1.0,
    "classes": [45, 43, 168, 165, 93, 86, 44, 41, 0, 32] + ([8, 9] if VM else []),   # VM 8/9: report separately
    "vm_arms": os.environ.get("VM_ARMS", "spikes,others").split(","),   # VM arms from the sets json (early / late too)
    "twobytwo": os.environ.get("TWOBYTWO", "") == "1",
    "conds": ("R", "Rev"), "res_conds": ("C1",) if VM else (),   # res_conds: baseline-only pass, residual sums only
    "n_pos": 8 if VM else 12, "n_tok": 1568 if VM else 2352, "d_model": 768, "n_heads": 12,
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "pair_sets": ROOT / "outputs/analysis/umt_pair_contrast/sets.csv",
    "sets_csv": ROOT / f"outputs/analysis/umt_layer_positional/sets_L{L}.csv",
    "extra_sets": ROOT / f"src/stage3_analysis/{TAG}_clock_downstream_sets.json",
    "out_dir": Path(os.environ.get("OUT_DIR", str(ROOT / f"outputs/analysis/{TAG}_clock_downstream_L{L}{RAW}"))),
    "max_clips": int(os.environ.get("MAX_CLIPS", 0)), "edit_batch": 16,
    "attr": RAW == "_raw" and os.environ.get("ATTR", "1") == "1", "attr_batch": int(os.environ.get("ATTR_BATCH", 8)),   # attribution: raw runs only
    "num_workers": int(os.environ.get("NUM_WORKERS", 4)), "save_every": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
if os.environ.get("CLASSES"):                                   # subset run, e.g. CLASSES=8,9 (use its own OUT_DIR)
    CFG["classes"] = [int(c) for c in os.environ["CLASSES"].split(",")]
CFG["val_json"] = Path(os.environ.get("VAL_JSON", str(Path(CFG["labels_path"]).parent / "validation.json")))


def clip_table(cfg):
    """Every val clip of the pair classes: clip_id, class_id, role (sel / test / none, from the pair sets)."""
    norm = lambda s: s.replace("[", "").replace("]", "")
    ids = {norm(k): int(v) for k, v in json.load(open(cfg["labels_path"])).items()}
    v = pd.DataFrame([{"clip_id": str(d["id"]), "class_id": ids[norm(d["template"])]}
                      for d in json.load(open(cfg["val_json"]))])
    v = v[v.class_id.isin(cfg["classes"])]
    s = pd.read_csv(cfg["pair_sets"], dtype={"clip_id": str}).drop_duplicates("clip_id").set_index("clip_id")
    v["role"] = v.clip_id.map(s.role).fillna("none")
    return v.sort_values("clip_id").head(cfg["max_clips"] or None).reset_index(drop=True)


def arm_rows(cfg):
    """[(arm, feats)]: baseline, batch_base, the positional selection arms, then clock (ramps + near-miss).
    VM: baseline, batch_base, spikes, others (sets json; VM has no positional selection sets)."""
    if cfg["model"] == "videomae":
        s = json.load(open(cfg["extra_sets"]))[str(cfg["layer"])]
        return [("baseline", []), ("batch_base", [])] + [(a, s[a]) for a in cfg["vm_arms"]]
    if cfg["twobytwo"]:                     # 2x2 (12/11/26): clock x builders vs neither; the interaction is the gate
        s = json.load(open(cfg["extra_sets"]))[str(cfg["layer"])]
        clock = s["clock"]
        return [("baseline", []), ("batch_base", []), ("clock", clock),
                ("builders", s["builders"]), ("both", clock + s["builders"])]
    import umt_layer_positional_ablation as pos
    arms = pos.arm_sets(dict(pos.CFG, layer=cfg["layer"], sets_csv=cfg["sets_csv"]))
    clock = json.load(open(cfg["extra_sets"]))[str(cfg["layer"])]["clock"]
    return [("baseline", []), ("batch_base", [])] + list(arms.items()) + [("clock", clock)]


def setup(cfg):
    """clock.setup (model, LAYER SAE + splice hook), plus per capture layer its SAE / dim_mean and a
    store-only output hook, and a store-only input hook on block attn_block's attention."""
    import umt_vtc_clock_ablation as clock
    from ToT_utils import MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    st, m = clock.setup(cfg), MODEL_REGISTRY[cfg["model"]]
    st.update(cap={}, down={}, attn_in=None, grad=False)     # grad=True: capture hooks keep the graph
    for c in cfg["capture"]:
        res = resolve_sae_checkpoint(cfg["model"], c, dataset_name="ssv2", sae_k=cfg["sae_k"], job_label=cfg["job_label"])
        sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=cfg["sae_k"])
        dm = torch.load(res["dim_mean_path"], weights_only=True).to(cfg["device"]).float()
        st["down"][c] = (sae, dm)
        m["layer_getter"](st["model"], c).register_forward_hook(
            lambda mod, i, o, c=c: st["cap"].__setitem__(
                c, (o[0] if isinstance(o, tuple) else o) if st["grad"] else (o[0] if isinstance(o, tuple) else o).detach()))
    blk = m["layer_getter"](st["model"], cfg["attn_block"])
    if cfg["model"] == "videomae":                               # LN'd attention input = layernorm_before output
        blk.layernorm_before.register_forward_hook(lambda mod, i, o: st.__setitem__("attn_in", o.detach()))
        st["attn_mod"] = blk.attention.attention
    else:
        blk.attn.register_forward_hook(lambda mod, i, o: st.__setitem__("attn_in", i[0].detach()))
        st["attn_mod"] = blk.attn
    st["mname"] = cfg["model"]
    return st


def down_profiles(st, c):
    """(B, 12, F) fp16 frame profiles of capture layer c's latents for the batch just run (encode only)."""
    from ToT_utils import gather_by_position
    sae, dm = st["down"][c]
    out = []
    for x in st["cap"][c].float():                               # (N, D) per row
        _, z = sae.encode(x - dm)
        out.append(gather_by_position(z, st["mname"]).sum(1).half())
    return torch.stack(out)


def scaled_qk(st, x, H):
    """Scaled queries and keys (B, N, D) of block attn_block. UMT: fused qkv; VM: separate query / key
    (both: q bias, no k bias)."""
    a = st["attn_mod"]
    if st["mname"] == "videomae":
        q = (x @ a.query.weight.float().T + a.q_bias.float()) * (x.shape[-1] // H) ** -0.5
        return q, x @ a.key.weight.float().T
    W = a.qkv.weight.float()
    return (x @ W[:W.shape[1]].T + a.q_bias.float()) * a.scale, x @ W[W.shape[1]:2 * W.shape[1]].T


def attn_frames(st, n_frames=12, H=12):
    """(B, H, T, T) attention mass from query position to key position, per head, from the stored LN'd input."""
    x = st["attn_in"].float()
    q, k = scaled_qk(st, x, H)
    B, N, _ = x.shape
    q, k = q.view(B, N, H, -1).transpose(1, 2), k.view(B, N, H, -1).transpose(1, 2)
    P, out = N // n_frames, torch.zeros(B, H, n_frames, n_frames, device=x.device)
    for f in range(n_frames):
        w = (q[:, :, f * P:(f + 1) * P] @ k.transpose(-1, -2)).softmax(-1)      # (B, H, P, N)
        out[:, :, f] = w.view(B, H, P, n_frames, P).sum((2, 4)) / P
    return out.half()


def attribute(st, cfg, pv, x, z0, tg):
    """Raw stream, unablated. tg = [(c, t)]: grad of target t's clip-summed activation at layer c w.r.t. layer-L
    tokens g, split over the layer-L dictionary: A[frame, f] = sum_tokens z_f (d_f . g); last column = remainder
    (x - z D, i.e. error + dim_mean) . g. Returns (len(tg), 12, F+1) float32 (first-order: activation x gradient)."""
    import umt_vtc_clock_ablation as clock
    from ToT_utils import gather_by_position
    out, rem, st["grad"] = [], x[0] - z0 @ st["D"], True
    try:
        for i in range(0, len(tg), cfg["attr_batch"]):
            part = tg[i:i + cfg["attr_batch"]]
            new = x.expand(len(part), -1, -1).clone().requires_grad_(True)
            with torch.enable_grad():
                clock.full_logits(st, pv, new)
                y = sum(st["down"][c][0].encode(st["cap"][c][b].float() - st["down"][c][1])[1][:, t].sum()
                        for b, (c, t) in enumerate(part))
                g, = torch.autograd.grad(y, new)
            for gb in g:
                a = torch.cat([z0 * (gb @ st["D"].T), (rem * gb).sum(-1, keepdim=True)], 1)
                out.append(gather_by_position(a, st["mname"]).sum(1))
    finally:
        st["grad"] = False
    return torch.stack(out) if out else torch.zeros(0, cfg["n_pos"], z0.shape[1] + 1, device=x.device)


def home_targets(cfg, cls):
    """[(capture layer, latent)] attribution targets whose home classes include cls (sets json attr_targets)."""
    tg = json.load(open(cfg["extra_sets"]))["attr_targets"]
    return [(c, int(t)) for c in cfg["capture"] for t, home in tg[str(c)].items() if cls in home]


def tubelet_res(st, cfg, x):
    """Baseline residual per position (frame / tubelet), mean over spatial tokens, at LAYER and each capture
    layer -> {layer: (T, D)}. Read before any edit batch overwrites st['cap']. Feeds the position-fraction ruler."""
    from ToT_utils import gather_by_position
    src = {cfg["layer"]: x[0], **{c: st["cap"][c][0] for c in cfg["capture"]}}
    return {l: gather_by_position(v.float(), st["mname"]).mean(1) for l, v in src.items()}


def one_clip(st, cfg, pv, arms, comb, cls):
    """Rows in arm_rows order: logits (R, 174), profiles {c: (R, T, F)}, attention (R, H, T, T);
    baseline extras: norms {layer: (N,)}, comb map (N,), attribution (K, T, F+1) for cls's home targets (raw),
    residual per position {layer: (T, D)}."""
    import umt_vtc_clock_ablation as clock
    with torch.no_grad():
        base, x, z0 = clock.clip_pass(st, pv)
        prof = {c: [down_profiles(st, c)] for c in cfg["capture"]}
        att, lg = [attn_frames(st, cfg["n_pos"])], [base[None]]
        res = tubelet_res(st, cfg, x)
        norms = {cfg["layer"]: x[0].norm(dim=-1).half()}
        norms.update({c: st["cap"][c][0].float().norm(dim=-1).half() for c in cfg["capture"]})
        comb_map = z0[:, comb].sum(1).half()
        attr = attribute(st, cfg, pv, x, z0, home_targets(cfg, cls)).half() if cfg["attr"] else None
        edits =[fs for a, fs in arms if a != "baseline"]
        for i in range(0, len(edits), cfg["edit_batch"]):
            new = torch.stack([clock.edited_stream(st, x, z0, st["D"], fs, cfg) for fs in edits[i:i + cfg["edit_batch"]]])
            lg.append(clock.full_logits(st, pv, new))
            att.append(attn_frames(st, cfg["n_pos"]))
            for c in cfg["capture"]:
                prof[c].append(down_profiles(st, c))
    return (torch.cat(lg).half(), {c: torch.cat(p) for c, p in prof.items()}, torch.cat(att), norms, comb_map, attr,
            res)


def open_store(cfg, cond, n, R, F=6144, base_only=False):
    """Memmaps for one condition (reopened in place if they exist -> resume); done flags (n,) bool.
    base_only (res_conds): residual per position only."""
    from numpy.lib.format import open_memmap
    N, T, H, lays = cfg["n_tok"], cfg["n_pos"], cfg["n_heads"], [cfg["layer"]] + cfg["capture"]
    shapes = {f"res_L{l}": (n, T, cfg["d_model"]) for l in lays}
    if not base_only:
        shapes.update({"logits": (n, R, 174), "attn": (n, R, H, T, T), "comb": (n, N)})
        shapes.update({f"prof_L{c}": (n, R, T, F) for c in cfg["capture"]})
        shapes.update({f"norm_L{l}": (n, N) for l in lays})
    if cfg["attr"] and not base_only:                            # K = home targets per clip (same for every class)
        K = len(home_targets(cfg, cfg["classes"][0]))
        shapes.update({"attr": (n, K, T, F + 1), "attr_tg": (n, K, 2)})
    shapes["done"] = (n,)
    store = {}
    for k, shp in shapes.items():
        p = cfg["out_dir"] / f"{k}_{cond}.npy"
        dt = {"done": np.bool_, "attr_tg": np.int16}.get(k, np.float32 if k.startswith("res_") else np.float16)
        store[k] = open_memmap(p, mode="r+") if p.exists() else open_memmap(p, mode="w+", dtype=dt, shape=shp)
        assert store[k].shape == shp, f"{p.name}: shape {store[k].shape} != {shp} (clip table changed?)"
    return store


def flush(store):
    for m in store.values():
        m.flush()


def write_clip(S, cfg, i, cls, out):
    """One clip's one_clip outputs into the memmaps at row i."""
    lg, prof, att, norms, cm, attr, _ = out
    S["logits"][i], S["attn"][i], S["comb"][i] = lg.cpu(), att.cpu(), cm.cpu()
    if cfg["attr"]:
        S["attr"][i], S["attr_tg"][i] = attr.cpu(), np.array(home_targets(cfg, cls), dtype=np.int16)
    for c, p in prof.items():
        S[f"prof_L{c}"][i] = p.cpu()
    for l, v in norms.items():
        S[f"norm_L{l}"][i] = v.cpu()


def run_cond(cfg, st, rows, arms, comb, cond, base_only=False):
    """Every not-yet-done clip under one condition -> the memmaps; done flags flushed every save_every.
    Loader / condition from umt_layer_positional_ablation (UMT: R + flip; VM: C1 shuffle in the dataset).
    base_only: one unedited forward per clip, residual per position only."""
    import umt_layer_positional_ablation as pos
    import umt_vtc_clock_ablation as clock
    S = open_store(cfg, cond, len(rows), len(arms), base_only=base_only)
    todo, t0, n = rows[~S["done"][:].astype(bool)], time.time(), 0
    for pv, i in pos.loader(cfg, st, todo, cond):
        i = int(i)
        if S["done"][i]:
            continue                                             # dataset skip-to-next landed on a done clip
        cls, pv = int(rows.class_id[i]), pos.condition(pv, cond).to(cfg["device"])
        if base_only:
            res = tubelet_res(st, cfg, clock.clip_pass(st, pv)[1])
        else:
            out = one_clip(st, cfg, pv, arms, comb, cls)
            write_clip(S, cfg, i, cls, out)
            res = out[-1]
        for l, v in res.items():
            S[f"res_L{l}"][i] = v.cpu()
        S["done"][i], n = True, n + 1
        if n % cfg["save_every"] == 0:
            flush(S)
            print(f"  {cond}: {n}/{len(todo)} clips, {(time.time() - t0) / n:.2f} s/clip", flush=True)
    flush(S)
    print(f"{cond}: {int(S['done'][:].sum())}/{len(rows)} clips done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main(cfg):
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    rows, arms = clip_table(cfg), arm_rows(cfg)
    comb = json.load(open(cfg["extra_sets"]))[str(cfg["layer"])]["comb"]
    rows.to_csv(cfg["out_dir"] / "clips.csv", index=False)
    pd.DataFrame([{"row": j, "arm": a, "k": len(f), "features": ",".join(map(str, f))}
                  for j, (a, f) in enumerate(arms)]).to_csv(cfg["out_dir"] / "arms.csv", index=False)
    print(f"LAYER {cfg['layer']} capture {cfg['capture']} | {len(rows)} clips "
          f"({(rows.role != 'sel').sum()} non-selection) | arms {[a for a, _ in arms]}", flush=True)
    st = setup(cfg)
    for cond in cfg["conds"]:
        run_cond(cfg, st, rows, arms, comb, cond)
    for cond in cfg["res_conds"]:                                # VM C1: residual per position only
        run_cond(cfg, st, rows, arms, comb, cond, base_only=True)


if __name__ == "__main__":
    main(CFG)
