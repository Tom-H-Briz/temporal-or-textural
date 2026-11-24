"""TimeSformer version of vm_layer_pe_characterise.py (24/11/26): per latent at L5 / L7 / L9, per-frame activation (8)
on a blank clip (full / frozen temporal PE), the 34 synthetic clips (SYNTH_ONLY=1, local) and UMT's 696 clip ids as
R / C / Rev. Frozen = frame 0's row of the separate time-embedding table on all 8 frames (spatial PE untouched).
Same npz keys as VM (C in place of C1). Patch tokens only (CLS dropped); TF is patch-major, so grouped by frame via
gather_by_position. Out: outputs/analysis/tf_layer_pe_characterise/{L{L}.npz, synth_L{L}.npz, clips.csv}.
Usage: [SYNTH_ONLY=1] [MAX_CLIPS=4] [MASSIVE_NORM=inf] python -u src/stage3_analysis/tf_layer_pe_characterise.py"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
CFG = dict(model="timesformer", layers=(5, 7, 9), n_pos=8, n_frames=8, bg_val=0.175,
           massive_norm=float(os.environ.get("MASSIVE_NORM", "inf")), synth_only=os.environ.get("SYNTH_ONLY") == "1",
           umt_clips=ROOT / "outputs/analysis/umt_layer_pe_characterise/clips.csv",   # same 696 ids as UMT / VM
           synth_dir=ROOT / "outputs/synthetic_motion/directional_clips/full_rate", max_clips=int(os.environ.get("MAX_CLIPS", 0)),
           video_dir=os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")), num_workers=int(os.environ.get("NUM_WORKERS", 4)),
           out=ROOT / "outputs/analysis/tf_layer_pe_characterise", device="cuda" if torch.cuda.is_available() else "cpu")


def load_all(cfg):
    """TF SSv2 model, its processor, and SAE + dim_mean per capture layer (trained eval threshold via load_sae_eval)."""
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    m, ck = MODEL_REGISTRY["timesformer"], CHECKPOINT_REGISTRY[("timesformer", "ssv2")]
    model = m["model_class"].from_pretrained(ck).to(cfg["device"]).eval()
    saes = {}
    for L in cfg["layers"]:
        r = resolve_sae_checkpoint("timesformer", L, dataset_name="ssv2", sae_k=64)   # TF-SSv2: job label = layer
        saes[L] = (load_sae_eval(r["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=64),
                   torch.load(r["dim_mean_path"], weights_only=True).to(cfg["device"]).float())
    return model, m["processor_class"].from_pretrained(ck), saes


def time_variant(te, cond):
    """(1, 8, 768) time table: full, or frozen = frame 0's row on all 8 frames (time removed; spatial PE untouched)."""
    return te if cond == "full" else te[:, :1].expand_as(te)


def embed(emb, pv, te):
    """HF TimesformerEmbeddings.forward (eval: dropout off, no resize) with time table te -> (1, 1 + 196*8, 768),
    patch-major (token = 1 + patch*8 + frame) as the encoder expects."""
    x, T, _ = emb.patch_embeddings(pv)                                    # (T, 196, D); 3rd return = grid width 14
    D = x.shape[-1]
    x = torch.cat((emb.cls_token.expand(x.size(0), -1, -1), x), dim=1) + emb.position_embeddings
    cls, x = x[:1, 0:1], x[:, 1:]                                         # CLS from frame 0, as HF does (batch 1)
    x = x.reshape(1, T, 196, D).permute(0, 2, 1, 3).reshape(196, T, D) + te
    return torch.cat((cls, x.reshape(1, 196 * T, D)), dim=1)


def blank(cfg, proc):
    """(1, 8, 3, 224, 224) grey clip, uint8-quantised as in the UMT probe, normalised with TF's own processor stats."""
    g, mu, sd = round(cfg["bg_val"] * 255) / 255.0, torch.tensor(proc.image_mean), torch.tensor(proc.image_std)
    return (torch.full((1, cfg["n_frames"], 3, 224, 224), g) - mu.view(1, 1, 3, 1, 1)) / sd.view(1, 1, 3, 1, 1)


def run_layers(model, saes, pv, te, cfg):
    """embed (time table te) -> layers 0..max(L). Per capture layer, on patch tokens grouped by frame (8, 196):
    (profile (8, F) ordinary tokens, same for massive tokens, n_massive (8,), token norm [max, p99] per frame (2, 8))."""
    tf, out = model.timesformer, {}
    with torch.no_grad():
        x = embed(tf.embeddings, pv.float().to(cfg["device"]), te.to(cfg["device"]))
        for i, layer in enumerate(tf.encoder.layer[:max(cfg["layers"]) + 1]):
            x = layer(x)
            x = x[0] if isinstance(x, tuple) else x
            if i in saes:
                out[i] = capture(saes[i], x[0, 1:], cfg)                  # CLS dropped
    return out


def capture(sae_dm, tok, cfg):
    """tok (1568, D) patch-major -> grouped (8, 196, .) by frame (gather_by_position, TF rule), same outputs as VM."""
    from ToT_utils import gather_by_position
    sae, dm = sae_dm
    nrm = tok.norm(dim=-1)
    ok = (nrm < cfg["massive_norm"]).float()
    _, z = sae.encode(tok - dm)
    prof = lambda w: gather_by_position(z * w[:, None], "timesformer").sum(1).cpu().numpy().astype(np.float16)
    n = gather_by_position(nrm, "timesformer")                            # (8, 196)
    stats = torch.stack([n.max(1).values, n.quantile(0.99, dim=1)]).cpu().numpy()
    return prof(ok), prof(1 - ok), gather_by_position(1 - ok, "timesformer").sum(1).cpu().numpy(), stats


def loader(ids, video_dir, cond, processor, cfg):
    """Batch-1 (pv, row index) TF loader (PerturbedKineticsDataset via umt_vision_pair_contrast): R or C."""
    from umt_vision_pair_contrast import clip_loader
    return clip_loader(list(zip(ids, range(len(ids)))), dict(cfg, video_dir=video_dir), "C" if cond == "C" else "R", processor)


def add(acc, key, out):
    """Append one input's per-layer outputs under key (same key names as the VM / UMT npz)."""
    for L, (p, pm, nm, ns) in out.items():
        for k, v in ((key, p), (f"{key}_massive", pm), (f"n_massive_{key}", nm), (f"norm_{key}", ns)):
            acc[L].setdefault(k, []).append(v)


def save(acc, cfg, prefix):
    cfg["out"].mkdir(parents=True, exist_ok=True)
    for L, d in acc.items():
        np.savez(cfg["out"] / f"{prefix}L{L}.npz", **{k: np.stack(v) for k, v in d.items()})


def main(cfg):
    model, proc, saes = load_all(cfg)
    te, acc = model.timesformer.embeddings.time_embeddings, {L: {} for L in cfg["layers"]}
    if cfg["synth_only"]:                                              # local: synth clips are not on Isambard
        names = sorted(p.stem for p in cfg["synth_dir"].glob("*.webm"))
        for pv, _ in loader(names, cfg["synth_dir"], "R", proc, cfg):
            add(acc, "synth", run_layers(model, saes, pv, te, cfg))
        return save(acc, cfg, "synth_"), print("synth clips:", names, flush=True)
    fr = os.environ.get("FROZEN_REAL") == "1"                          # 24/11: real clips under frozen time table only
    for c in () if fr else ("full", "frozen"):
        add(acc, f"blank_{c}", run_layers(model, saes, blank(cfg, proc), time_variant(te, c), cfg))
    rows = pd.read_csv(cfg["umt_clips"], dtype={"clip_id": str}).head(cfg["max_clips"] or None)
    te_c = time_variant(te, "frozen") if fr else te
    for cond in ("R_frozen", "Rev_frozen") if fr else ("R", "C", "Rev"):    # Rev = the R clip, all 8 frames flipped
        for pv, i in loader(rows.clip_id.tolist(), cfg["video_dir"], cond, proc, cfg):
            pv = pv.flip(1) if cond.startswith("Rev") else pv
            add(acc, cond, run_layers(model, saes, pv, te_c, cfg))
            acc[cfg["layers"][0]].setdefault(f"row_{cond}", []).append(int(i))
            if cond == "R":                                            # raw TF top-1, to gate on R-correct
                with torch.no_grad():
                    acc[cfg["layers"][0]].setdefault("pred_R", []).append(int(model(pixel_values=pv.to(cfg["device"])).logits.argmax()))
        print(f"{cond}: {len(rows)} clips done", flush=True)
    p = "frozen_" if fr else ""
    save(acc, cfg, p), rows.to_csv(cfg["out"] / f"{p}clips.csv", index=False)


if __name__ == "__main__":
    main(CFG)
