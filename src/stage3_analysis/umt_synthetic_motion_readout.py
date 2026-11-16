"""
Synthetic-motion readout (16/11/26): the motion_vid_generator clips (ball / pyramid x 8 directions x 2 speeds
+ 2 static) through plain UMT (no splice, no ablation). Per clip: VTC logits (174) + top-10 classes; at each SAE
layer, all 6,144 latents per frame (12, F) and per-token sparse codes (12, 196, K) to map where features fire.
Usage: python -u src/stage3_analysis/umt_synthetic_motion_readout.py   (run the generator first)
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

CFG = {
    "model": "umt", "layers": [5, 7, 9], "sae_k": 64, "job_label": "7ep",
    "k_store": 128,  # per-token latents kept (SAE uses a trained threshold, so active count can exceed 64)
    "clip_dir": Path(os.environ.get("CLIP_DIR", str(ROOT / "outputs/synthetic_motion/directional_clips"))),
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "out_dir": Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/umt_synthetic_motion_readout"))),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def setup(cfg):
    """UMT (VTC head) + per layer its SAE (trained threshold) and dim_mean, with a store-only output hook."""
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    m, dev = MODEL_REGISTRY[cfg["model"]], cfg["device"]
    model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[(cfg["model"], "ssv2")]).to(dev).eval()
    st = {"model": model, "sae": {}, "x": {}}
    for L in cfg["layers"]:
        res = resolve_sae_checkpoint(cfg["model"], L, dataset_name="ssv2", sae_k=cfg["sae_k"], job_label=cfg["job_label"])
        sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], dev, sae_k=cfg["sae_k"])
        dm = torch.load(res["dim_mean_path"], weights_only=True).to(dev).float()
        st["sae"][L] = (sae, dm)
        m["layer_getter"](model, L).register_forward_hook(
            lambda mod, i, o, L=L: st["x"].__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, m["cls_offset"]:].detach()))
        print(f"L{L} SAE {Path(res['sae_path']).name} threshold {float(sae.running_threshold):.3f}", flush=True)
    return st


def read_clip(st, cfg, pv):
    """One clip (1, T, C, H, W) -> logits (174,), and per layer: frame profile (12, F) and sparse
    per-token codes idx / val (12, 196, k_store), plus the max active-latent count per token."""
    from ToT_utils import gather_by_position
    with torch.no_grad():
        logits = st["model"](pixel_values=pv.to(cfg["device"])).logits[0].float().cpu()
        out = {}
        for L, (sae, dm) in st["sae"].items():
            _, z = sae.encode(st["x"][L][0].float() - dm)               # (N, F), N = 12 x 196
            zp = gather_by_position(z, cfg["model"])                     # (12, 196, F), frame-major
            val, idx = zp.topk(cfg["k_store"], dim=-1)
            # Per-token maps (12, 196): massive high-norm tokens fire 600+ latents, so the summed profile is
            # dominated by them; n_active <= k_store marks tokens whose sparse codes are complete
            norm = gather_by_position(st["x"][L][0].float().norm(dim=-1), cfg["model"])
            n_active = (zp != 0).sum(-1)
            out[L] = {"profile": zp.sum(1).half().cpu(), "idx": idx.short().cpu(), "val": val.half().cpu(),
                      "norm": norm.half().cpu(), "n_active": n_active.short().cpu(),
                      "max_active": int(n_active.max())}
    return logits, out


def top_classes(logits, cfg, meta, k=10):
    """Long table: one row per clip x rank with class id, name and softmax probability."""
    names = {int(v): n for n, v in json.load(open(cfg["labels_path"])).items()}
    probs = torch.softmax(logits, dim=-1)
    rows = []
    for i, m in enumerate(meta):
        p, c = probs[i].topk(k)
        rows += [{"clip": m["name"], "rank": r + 1, "class_id": int(c[r]), "class_name": names[int(c[r])],
                  "prob": float(p[r]), "logit": float(logits[i, c[r]])} for r in range(k)]
    return pd.DataFrame(rows)


def main(cfg):
    meta = json.load(open(cfg["clip_dir"] / "metadata.json"))
    batch = torch.load(cfg["clip_dir"] / "all_umt_clips.pt")              # (N, T, C, H, W)
    st, out_dir = setup(cfg), cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    reads = [read_clip(st, cfg, batch[i:i + 1]) for i in range(len(meta))]
    logits = torch.stack([lg for lg, _ in reads])
    np.save(out_dir / "logits.npy", logits.numpy())
    top_classes(logits, cfg, meta).to_csv(out_dir / "top_classes.csv", index=False)
    for L in cfg["layers"]:
        for key in ("profile", "idx", "val", "norm", "n_active"):
            np.save(out_dir / f"{key}_L{L}.npy", torch.stack([r[L][key] for _, r in reads]).numpy())
        print(f"L{L} max active latents per token: {max(r[L]['max_active'] for _, r in reads)}", flush=True)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main(CFG)
