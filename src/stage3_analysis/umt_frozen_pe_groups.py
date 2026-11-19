"""Frozen-PE ablation of the order-sensitive / insensitive latent groups (19/11/26). Spliced, UMT L5 / L7 / L9.
Per clip and layer: a full-PE and a frozen-PE (frame 0's PE on all frames) spliced DFA pass -> signed DFA (F,),
per-frame activation over ordinary tokens (12, F; massive = residual norm >= 200), 174 logits. Then patched logits:
full-PE codes with one group's latents swapped for their frozen-PE values (arm "all" = every latent = frozen run).
Groups: umt_order_groups.json (umt_order_sensitivity_mass.py). Clips: the same 4 x 174 R clips as
umt_layer_pe_characterise. Out: outputs/analysis/umt_frozen_pe_groups/L{L}.npz. GPU (Isambard).
Usage: [MAX_CLIPS=4] [LAYERS=5,7,9] python -u src/stage3_analysis/umt_frozen_pe_groups.py"""
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
CFG = {"layers": [int(x) for x in os.environ.get("LAYERS", "5,7,9").split(",")], "massive_norm": 200.0,
       "groups": Path(__file__).with_name("umt_order_groups.json"), "out": ROOT / "outputs/analysis/umt_frozen_pe_groups",
       "n_per_class": 4, "max_clips": int(os.environ.get("MAX_CLIPS", 0)), "num_workers": int(os.environ.get("NUM_WORKERS", 4)),
       "device": "cuda" if torch.cuda.is_available() else "cpu"}


@contextmanager
def pe_swapped(eng, cond):
    """The encoder's sinusoid PE (a plain tensor attribute, not a parameter) swapped for pe_variant(cond), restored after."""
    from umt_pe_probe import pe_variant
    v = eng._model.vision.encoder
    orig = v.pos_embed
    v.pos_embed = pe_variant(orig, cond)
    try:
        yield
    finally:
        v.pos_embed = orig


def norm_hook(st, cut):
    """Prepended layer-L hook (sees the raw residual, before the splice): st['ok'] = ordinary-token mask (N,)."""
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        st["ok"] = (h[0].norm(dim=-1) < cut).float()
    return hook


def one_pass(eng, pv, cls, st):
    """Spliced DFA pass (+ a no-grad pass for token codes): signed DFA (F,), ordinary-token per-frame activation
    (12, F) float16, logits (174,), token codes z (N, F) kept on the GPU for patching."""
    r = eng.run_pixels(pv, cls)
    z = eng.get_z_pixels(pv)                                     # same forward; also refreshes st['ok']
    act = (z * st["ok"][:, None]).reshape(12, 196, -1).sum(1).cpu().numpy().astype(np.float16)
    return r.signed_feature_summary.numpy(), act, r.all_logits.numpy(), z


def patched_logits(eng, pv, z_full, z_frz, groups):
    """(arms, 174) logits: full-PE codes with each group's latents set to their frozen-PE values (spliced, no grad)."""
    out = []
    for g in groups.values():
        z = z_full.clone()
        z[:, g] = z_frz[:, g]
        eng._z_override = z
        try:
            with torch.no_grad():
                out.append(eng._model(pixel_values=pv).logits[0].float().cpu().numpy())
        finally:
            eng._z_override = None
    return np.stack(out)


def run_layer(L, rows, cfg, groups):
    """Every clip at layer L -> L{L}.npz: {full, frozen}_{dfa, act, logits}, patched (clips, arms, 174), arms, row."""
    from dfa_engine import DFAEngine
    from ToT_utils import MODEL_REGISTRY, resolve_sae_checkpoint
    from umt_l3_pe_lineage import CFG as LCFG
    from umt_mlm_feature_contrast import clip_loader
    r, acc, st = resolve_sae_checkpoint("umt", L, dataset_name="ssv2", sae_k=64, job_label="7ep"), {}, {}
    with DFAEngine("umt", r["sae_path"], r["dim_mean_path"], layer=L, device=cfg["device"], sae_k=64) as eng:
        MODEL_REGISTRY["umt"]["layer_getter"](eng._model, L).register_forward_hook(norm_hook(st, cfg["massive_norm"]), prepend=True)
        for pv, i in clip_loader(rows, dict(LCFG, num_workers=cfg["num_workers"]), "R"):
            pv, cls = pv.to(cfg["device"]), int(rows.class_id.iloc[int(i)])
            full = one_pass(eng, pv, cls, st)
            with pe_swapped(eng, "frozen"):
                frz = one_pass(eng, pv, cls, st)
            for cond, res in (("full", full), ("frozen", frz)):
                for k, v in zip(("dfa", "act", "logits"), res[:3]):
                    acc.setdefault(f"{cond}_{k}", []).append(v)
            acc.setdefault("patched", []).append(patched_logits(eng, pv, full[3], frz[3], groups))
            acc.setdefault("row", []).append(int(i))
    np.savez(cfg["out"] / f"L{L}.npz", arms=np.array(list(groups)), **{k: np.stack(v) for k, v in acc.items()})


def main(cfg):
    from umt_layer_pe_characterise import clip_rows
    rows = clip_rows(cfg)                                        # same R clips as the characterisation run
    cfg["out"].mkdir(parents=True, exist_ok=True)
    rows.to_csv(cfg["out"] / "clips.csv", index=False)
    sets = json.loads(cfg["groups"].read_text())
    for L in cfg["layers"]:
        groups = {**sets[str(L)], "all": list(range(6144))}      # "all" must reproduce the frozen logits (check)
        run_layer(L, rows, cfg, groups)
        print(f"L{L}: {len(rows)} clips done", flush=True)


if __name__ == "__main__":
    main(CFG)
