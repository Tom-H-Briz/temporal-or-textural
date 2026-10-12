"""UMT vision-side pair contrast (Stage 1), matched to the text-side pairs: move up (45) vs
move down (43), vision L9 SAE with the TRAINED threshold (DFAEngine, post 12/10/26 fix).

Extract (GPU): every val clip of both classes, three conditions:
  R   real clip
  C   native-frame shuffle pre-sampling (UMT/TF convention, seed from clip id)
  Rev the 12 sampled frames in reverse order (an "up" clip played backwards looks "down")
Per clip and condition: token-level sparse SAE codes z and contrast DFA = dz-gradient of
[logit(up) - logit(down)] x z, both (2352 tokens = 12 frames x 196 patches, 6144 latents),
plus the 174 VTC logits. Aggregation is a separate script (two-stage persistence).
Usage: python src/stage3_analysis/umt_vision_pair_contrast.py   (MAX_CLIPS=8 for smoke)
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

CFG = {
    "pair":            (45, 43),        # (A, B) = (moving something up, moving something down)
    "conditions":      ("R", "C", "Rev"),
    "layer":           9,
    "sae_path":        ROOT / "outputs/sae/sae_umt_ssv2_k64_x8_l9_job7ep_best.pt",
    "dim_mean_path":   ROOT / "outputs/sae/umt_ssv2_layer9_dim_mean.pt",
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "max_clips":       int(os.environ.get("MAX_CLIPS", 0)),
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":     4,
    "out_dir":         ROOT / "outputs/analysis/umt_vision_pair_contrast",
}


def pair_clips(cfg):
    """All val clips of the pair's two classes, sorted by id: [(clip_id, class_id)].
    Same template normalisation as umt_vision_text_bridge.t1_clips."""
    norm = lambda s: s.lower().replace("[", "").replace("]", "")
    key2id = {norm(k): int(v) for k, v in json.load(open(cfg["labels_path"])).items()}
    rows = sorted((c["id"], key2id[norm(c["template"])]) for c in json.load(open(cfg["validation_path"]))
                  if key2id.get(norm(c["template"])) in cfg["pair"])
    return rows[: cfg["max_clips"]] if cfg["max_clips"] else rows


def clip_loader(rows, cfg, condition):
    """Batch-1 loader of pixel_values (1, 12, C, H, W) with UMT's own sampler + preprocessing,
    the same dataset class as the perturb/R pipeline. Rev = R frames flipped in time, so the
    two conditions see identical frames in opposite order."""
    from ToT_utils import MODEL_REGISTRY, get_frame_sampler
    from perturb_accuracy_tf_kinetics import PerturbedKineticsDataset
    from umt_wrapper import UMTProcessor
    ds = PerturbedKineticsDataset(
        [Path(cfg["video_dir"]) / f"{i}.webm" for i, _ in rows], [i for i, _ in rows],
        [c for _, c in rows], UMTProcessor(), MODEL_REGISTRY["umt"]["num_frames"],
        get_frame_sampler("ssv2", MODEL_REGISTRY["umt"]), "C" if condition == "C" else "R")
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["num_workers"])


def apply_condition(pixel_values, condition):
    """Rev: reverse the time axis (dim 1). R and C are handled by the dataset."""
    return pixel_values.flip(1) if condition == "Rev" else pixel_values


def contrast_dfa(engine, pixel_values, a, b):
    """One clip through DFAEngine's L9 SAE splice; backward from logit(a) - logit(b).
    Returns VTC logits (174,), z (2352, 6144) and DFA = grad x z (2352, 6144), signed so
    positive = pushes toward class a. Mirrors DFAEngine.run_pixels, contrast target only."""
    engine._z = None
    logits = engine._model(pixel_values=pixel_values).logits.squeeze(0)
    z = engine._z
    if z is None:
        raise RuntimeError("DFA hook did not fire")
    (logits[a] - logits[b]).backward()
    dfa = (z.grad * z).detach()
    out = logits.detach().float().cpu(), z.detach(), dfa
    engine._z = None
    return out


def to_sparse(z, dfa):
    """COO over active entries (z > 0; DFA is zero wherever z is): token (int16), latent
    (int16), z (fp16), dfa (fp32). ~2352 x 66 entries per clip at the trained threshold."""
    tok, lat = torch.nonzero(z > 0, as_tuple=True)
    return dict(tok=tok.to(torch.int16).cpu().numpy(), lat=lat.to(torch.int16).cpu().numpy(),
                z=z[tok, lat].half().cpu().numpy(), dfa=dfa[tok, lat].float().cpu().numpy())


def extract(cfg):
    """Stage 1 raw data: sparse/{cond}/{clip_id}.npz + per_clip.parquet (logits, ids)."""
    from dfa_engine import DFAEngine
    a, b = cfg["pair"]
    rows, out, recs = pair_clips(cfg), cfg["out_dir"], []
    print(f"{len(rows)} clips  (A={a}: {sum(c == a for _, c in rows)}, B={b}: {sum(c == b for _, c in rows)})")
    with DFAEngine("umt", cfg["sae_path"], cfg["dim_mean_path"], layer=cfg["layer"], device=cfg["device"]) as eng:
        print(f"SAE threshold {float(eng._sae.running_threshold):.4f} (trained)")
        for cond in cfg["conditions"]:
            (out / "sparse" / cond).mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            for (cid, cls), (pv, _) in zip(rows, clip_loader(rows, cfg, cond)):
                pv = apply_condition(pv, cond).to(cfg["device"])
                logits, z, dfa = contrast_dfa(eng, pv, a, b)
                np.savez_compressed(out / "sparse" / cond / f"{cid}.npz", **to_sparse(z, dfa))
                recs.append(dict(clip_id=cid, true_class=cls, condition=cond, logits=logits.numpy(),
                                 margin_ab=float(logits[a] - logits[b]), pred=int(logits.argmax()),
                                 l0_per_token=float((z > 0).sum(1).float().mean())))
            print(f"  {cond}: {len(rows)} clips  {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(recs).to_parquet(out / "per_clip.parquet", index=False)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # match the verified fp32 paths
    torch.backends.cudnn.allow_tf32 = False
    cfg = dict(CFG)
    if cfg["max_clips"]:                              # smoke never overwrites the real run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"device={cfg['device']}  max_clips={cfg['max_clips'] or 'all'}  out={cfg['out_dir']}")
    extract(cfg)


if __name__ == "__main__":
    main()
