"""
PE probe (16/11/26): do UMT's L5 timing features fire on nothing, and is that driven by the positional encoding?
UMT's PE is a fixed sinusoid over the 2352 token positions (frame-major), as in VideoMAE. Inputs: a blank grey
clip, static ball / pyramid, one moving clip (synthetic suite). PE conditions: full / none (zeroed) / frozen
(frame 0's PE on all 12 frames: space kept, time removed) / reversed (frame order flipped). Per L5 latent:
per-frame activation summed over ordinary tokens (massive high-norm tokens excluded) -> pe_probe.parquet.
Usage: uv run python -u src/stage3_analysis/umt_pe_probe.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

SETS = json.load(open(ROOT / "src/stage3_analysis/umt_clock_downstream_sets.json"))["5"]
CFG = {
    "model": "umt", "layer": 5, "sae_k": 64, "job_label": "7ep",
    "massive_norm": 200.0,   # ordinary tokens ~30, massive ~450-700 at L5 (16/11 readout)
    "clip_dir": ROOT / "outputs/synthetic_motion/directional_clips",
    "clips": ["ball_static_center", "pyramid_static_center", "ball_dir_090_fast_center"],
    "bg_val": 0.175, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225),
    "pe_conds": ["full", "none", "frozen", "reversed"],
    "watch": {"clock": SETS["clock"], "coherent_ramp": [5216, 515], "builders": [5135, 4638, 186, 5519]},
    "out_dir": Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/umt_pe_probe"))),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def inputs(cfg):
    """{name: (1, T, C, H, W)}: blank grey clip (same normalisation as the suite) + the chosen synthetic clips."""
    mean = torch.tensor(cfg["mean"]).view(1, 1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 1, 3, 1, 1)
    grey = round(cfg["bg_val"] * 255) / 255.0                              # uint8-quantised like the suite
    out = {"blank": (torch.full((1, 12, 3, 224, 224), grey) - mean) / std}
    for n in cfg["clips"]:
        out[n] = torch.load(cfg["clip_dir"] / "umt_12f" / f"{n}.pt").unsqueeze(0)
    return out


def pe_variant(pe, cond):
    """(1, 2352, 768) PE under one condition; frames are blocks of 196 tokens (frame-major)."""
    p = pe.reshape(12, 196, -1)
    if cond == "none":
        p = torch.zeros_like(p)
    elif cond == "frozen":
        p = p[:1].expand(12, -1, -1)
    elif cond == "reversed":
        p = p.flip(0)
    elif cond.startswith("frozen"):                       # 25/11: frozenK = frame K's PE on all 12 frames
        k = int(cond[6:])
        p = p[k:k + 1].expand(12, -1, -1)
    return p.reshape(1, 2352, -1)


def run_l5(model, sae, dm, pv, pe, cfg):
    """patch_embed + pe -> blocks 0..L -> SAE codes. Returns profile (12, F) over ordinary tokens and the
    count of massive tokens per frame (12,). Same path as umt_bridge_phase3.tower_to_layer, PE swapped."""
    v = model.vision.encoder
    with torch.no_grad():
        x = v.patch_embed(pv.permute(0, 2, 1, 3, 4).float().to(cfg["device"])) + pe.to(cfg["device"])
        for blk in v.blocks[:cfg["layer"] + 1]:
            x = blk(x)
        x = x[0]
        _, z = sae.encode(x - dm)
        ok = (x.norm(dim=-1) < cfg["massive_norm"]).float()                      # (2352,)
        prof = (z * ok[:, None]).reshape(12, 196, -1).sum(1)                       # (12, F)
        n_massive = (1 - ok).reshape(12, 196).sum(1)
    return prof.cpu().numpy(), n_massive.cpu().numpy()


def main(cfg):
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    m = MODEL_REGISTRY[cfg["model"]]
    model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[(cfg["model"], "ssv2")]).to(cfg["device"]).eval()
    res = resolve_sae_checkpoint(cfg["model"], cfg["layer"], dataset_name="ssv2", sae_k=cfg["sae_k"], job_label=cfg["job_label"])
    sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=cfg["sae_k"])
    dm = torch.load(res["dim_mean_path"], weights_only=True).to(cfg["device"]).float()
    pe, rows, profs = model.vision.encoder.pos_embed, [], {}
    for name, pv in inputs(cfg).items():
        for cond in cfg["pe_conds"]:
            prof, nm = run_l5(model, sae, dm, pv, pe_variant(pe, cond), cfg)
            profs[f"{name}|{cond}"] = prof.astype(np.float16)
            rows.append(dict(input=name, pe=cond, n_massive=nm.tolist(), n_latents_on=int((prof.sum(0) > 0).sum())))
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    np.savez(cfg["out_dir"] / "profiles.npz", **profs)                               # key "input|pe": (12, F)
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "runs.csv", index=False)
    print(pd.DataFrame(rows).to_string(), flush=True)


if __name__ == "__main__":
    main(CFG)
