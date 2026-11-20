"""VideoMAE version of umt_layer_pe_characterise.py (20/11/26): per latent at L5 / L7 / L9, per-tubelet activation (8)
on a blank clip (full / frozen PE), the 34 synthetic clips (SYNTH_ONLY=1, local) and UMT's 696 clip ids as R / C1 / Rev.
Same npz keys as UMT (C1 in place of C), plus pred_R and row_<cond> (L5 npz only; raw VM top-1, to gate on R-correct
against clips.csv class_id) and norm_<key> (token-norm max / p99 per tubelet:
looking for massive tokens; MASSIVE_NORM sets the cut, default none).
Out: outputs/analysis/vm_layer_pe_characterise/{L{L}.npz, synth_L{L}.npz, clips.csv}. GPU (Isambard), synth on CPU.
Usage: [SYNTH_ONLY=1] [MAX_CLIPS=4] [MASSIVE_NORM=inf] python -u src/stage3_analysis/vm_layer_pe_characterise.py"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
CFG = dict(layers=(5, 7, 9), n_pos=8, n_frames=16, bg_val=0.175, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
           massive_norm=float(os.environ.get("MASSIVE_NORM", "inf")), synth_only=os.environ.get("SYNTH_ONLY") == "1",
           umt_clips=ROOT / "outputs/analysis/umt_layer_pe_characterise/clips.csv",   # same 696 ids as UMT
           synth_dir=ROOT / "outputs/synthetic_motion/directional_clips/full_rate", max_clips=int(os.environ.get("MAX_CLIPS", 0)),
           video_dir=os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")), num_workers=int(os.environ.get("NUM_WORKERS", 4)),
           out=ROOT / "outputs/analysis/vm_layer_pe_characterise", device="cuda" if torch.cuda.is_available() else "cpu")


def load_all(cfg):
    """VM SSv2 model, its processor, and SAE + dim_mean per capture layer (trained eval threshold via load_sae_eval)."""
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    m, ck = MODEL_REGISTRY["videomae"], CHECKPOINT_REGISTRY[("videomae", "ssv2")]
    model = m["model_class"].from_pretrained(ck).to(cfg["device"]).eval()
    saes = {}
    for L in cfg["layers"]:
        r = resolve_sae_checkpoint("videomae", L, dataset_name="ssv2", sae_k=64, job_label="7ep")
        saes[L] = (load_sae_eval(r["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=64),
                   torch.load(r["dim_mean_path"], weights_only=True).to(cfg["device"]).float())
    return model, m["processor_class"].from_pretrained(ck), saes


def pe_variant(pe, cond, n_pos=8):
    """(1, 1568, 768) PE: full, or frozen = tubelet 0's PE on all 8 tubelets (space kept, time removed)."""
    return pe if cond == "full" else pe.reshape(n_pos, 196, -1)[:1].expand(n_pos, -1, -1).reshape(1, -1, pe.shape[-1])


def blank(cfg):
    """(1, 16, 3, 224, 224) grey clip, uint8-quantised and normalised as in the UMT probe."""
    g = round(cfg["bg_val"] * 255) / 255.0
    return (torch.full((1, cfg["n_frames"], 3, 224, 224), g) - torch.tensor(cfg["mean"]).view(1, 1, 3, 1, 1)) / torch.tensor(cfg["std"]).view(1, 1, 3, 1, 1)


def run_layers(model, saes, pv, pe, cfg):
    """patch embed + pe -> layers 0..max(L). Per capture layer: (profile (8, F) ordinary tokens, same for massive
    tokens, n_massive (8,), token norm [max, p99] per tubelet (2, 8))."""
    v, P, out = model.videomae, cfg["n_pos"], {}
    with torch.no_grad():
        x = v.embeddings.patch_embeddings(pv.float().to(cfg["device"])) + pe.to(cfg["device"])
        for i, layer in enumerate(v.encoder.layer[:max(cfg["layers"]) + 1]):
            x = layer(x)
            x = x[0] if isinstance(x, tuple) else x
            if i in saes:
                sae, dm = saes[i]
                nrm = x[0].norm(dim=-1)
                ok = (nrm < cfg["massive_norm"]).float()
                _, z = sae.encode(x[0] - dm)
                prof = lambda w: (z * w[:, None]).reshape(P, 196, -1).sum(1).cpu().numpy().astype(np.float16)
                n = nrm.reshape(P, 196)
                stats = torch.stack([n.max(1).values, n.quantile(0.99, dim=1)]).cpu().numpy()
                out[i] = (prof(ok), prof(1 - ok), (1 - ok).reshape(P, 196).sum(1).cpu().numpy(), stats)
    return out


def loader(ids, video_dir, cond, processor, cfg):
    """Batch-1 (pv, row index) VM loader (PerturbedSSv2Dataset via umt_vision_pair_contrast): R or C1."""
    from umt_vision_pair_contrast import clip_loader
    return clip_loader(list(zip(ids, range(len(ids)))), dict(cfg, model="videomae", video_dir=video_dir),
                       "C1" if cond == "C1" else "R", processor)


def add(acc, key, out):
    """Append one input's per-layer outputs under key (same key names as the UMT npz)."""
    for L, (p, pm, nm, ns) in out.items():
        for k, v in ((key, p), (f"{key}_massive", pm), (f"n_massive_{key}", nm), (f"norm_{key}", ns)):
            acc[L].setdefault(k, []).append(v)


def save(acc, cfg, prefix):
    cfg["out"].mkdir(parents=True, exist_ok=True)
    for L, d in acc.items():
        np.savez(cfg["out"] / f"{prefix}L{L}.npz", **{k: np.stack(v) for k, v in d.items()})


def main(cfg):
    model, proc, saes = load_all(cfg)
    pe, acc = model.videomae.embeddings.position_embeddings, {L: {} for L in cfg["layers"]}
    if cfg["synth_only"]:                                              # local: synth clips are not on Isambard
        names = sorted(p.stem for p in cfg["synth_dir"].glob("*.webm"))
        for pv, _ in loader(names, cfg["synth_dir"], "R", proc, cfg):
            add(acc, "synth", run_layers(model, saes, pv, pe, cfg))
        return save(acc, cfg, "synth_"), print("synth clips:", names, flush=True)
    for c in ("full", "frozen"):
        add(acc, f"blank_{c}", run_layers(model, saes, blank(cfg), pe_variant(pe, c, cfg["n_pos"]), cfg))
    rows = pd.read_csv(cfg["umt_clips"], dtype={"clip_id": str}).head(cfg["max_clips"] or None)
    for cond in ("R", "C1", "Rev"):                                     # Rev = the R clip, all 16 frames flipped
        for pv, i in loader(rows.clip_id.tolist(), cfg["video_dir"], cond, proc, cfg):
            pv = pv.flip(1) if cond == "Rev" else pv
            add(acc, cond, run_layers(model, saes, pv, pe, cfg))
            acc[cfg["layers"][0]].setdefault(f"row_{cond}", []).append(int(i))
            if cond == "R":                                            # raw VM top-1, to gate on R-correct
                with torch.no_grad():
                    acc[cfg["layers"][0]].setdefault("pred_R", []).append(int(model(pixel_values=pv.to(cfg["device"])).logits.argmax()))
        print(f"{cond}: {len(rows)} clips done", flush=True)
    save(acc, cfg, ""), rows.to_csv(cfg["out"] / "clips.csv", index=False)


if __name__ == "__main__":
    main(CFG)
