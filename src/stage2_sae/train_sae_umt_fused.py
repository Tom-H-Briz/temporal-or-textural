"""Train BatchTopK SAEs on UMT's FUSED text states (text-encoder layers 9-11), L9 and L10.

Fused states = the 174 cached templates' layer-0-8 outputs run through fusion layers
against a real clip's vision tokens — the only place text meets video (cross-attention
K/V from the vision tower, queries from text). Training data: all-174 manifest
(6,705 clips x 174 templates), all real (non-pad) tokens, streamed per clip (a full
state cache would be ~45 GB/layer). Fusion runs fp32: the verified xbert-matching
path (fp16/TF32 shifts the VTM self-check to ~1e-3).
The existing vision-tower SAEs (sae_umt_ssv2_k64_x8_l{5,7,9}) are untouched.
Modes (MODE env var):
  dim_mean -> per-dim mean of real-token fused states at SAE_LAYER over
              DIM_MEAN_CLIPS train-split clips -> umt_ssv2_fusedtext_layer{L}_dim_mean.pt
  train    -> k64 x8 alpha 0.03 aux loss, SAE_EPOCHS epochs, seeded clip split
              (VAL_FRACTION) -> sae_umt_ssv2_fusedtext_k64_x8_l{L}_job{label}[_best].pt.
              Non-fatal epilogue: VTM 2-way pair accuracy, raw vs SAE-spliced.
  eval     -> splice gate + null-video verb-signature gate (verb-pair latent deltas
              sign-consistent across nulls) on the trained best checkpoint.
Local smoke: SAE_LAYER=10 SAE_JOB_LABEL=smoke SAE_EPOCHS=2 MAX_CLIPS=20 MODE=train \
  uv run --with easydict --with "timm==0.4.12" python src/stage2_sae/train_sae_umt_fused.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src")]
from sae import BatchTopKSAE
from sae.losses import top_k_auxiliary_loss
from stage3_analysis.umt_fusion_layer_scan import _fusion_layer_attn, make_frames, synthetic_videos
from stage3_analysis.umt_text_geometry import CFG as GEOM_CFG
from stage3_analysis.umt_vtm_gate import (
    _kv, _lin, load_vtm, test2_pairs, vision_and_vtc,
)
from umt_wrapper import CFG as UMT_CFG, UMTClassifier, UMTProcessor
from ToT_utils import MODEL_REGISTRY, get_frame_sampler
from perturb_accuracy_tf_kinetics import PerturbedKineticsDataset

CFG = {
    "mode":            os.environ.get("MODE", "train"),
    "layer":           int(os.environ.get("SAE_LAYER", 10)),  # 9 = first fusion, 10 = gate winner
    "manifest_path":   os.environ.get("MANIFEST_PATH", str(ROOT / "outputs/manifests/manifest_ssv2_all174_cap40.json")),
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "max_clips":       int(os.environ.get("MAX_CLIPS", 0)),       # >0 = smoke (splits + names unaffected)
    "dim_mean_clips":  int(os.environ.get("DIM_MEAN_CLIPS", 1000)),
    "epochs":          int(os.environ.get("SAE_EPOCHS", 7)),
    "val_fraction":    float(os.environ.get("VAL_FRACTION", 0.12)),
    "split_seed":      42,
    "batch_size":      8,
    "num_workers":     4,
    "lr":              1e-4,
    "job_label":       os.environ.get("SAE_JOB_LABEL", "7ep"),
    "sae_k":           64,
    "expansion":       8,
    "alpha":           0.03,
    "splice_cap":      int(os.environ.get("SPLICE_CAP", 300)),
    "resume_from":     os.environ.get("RESUME_FROM"),
    "wandb_project":   "temporal-or-textural",
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}


def build_split(cfg):
    """[(clip_id, class_id)] from the all-174 manifest (class via labels.json
    template->id), optionally smoke-capped, seeded shuffle, val fraction split."""
    labels = json.load(open(cfg["labels_path"]))
    rows = [(c["id"], int(labels[c["template"]]))
            for c in json.load(open(cfg["manifest_path"]))["all"]]
    if cfg["max_clips"]:
        rows = rows[: cfg["max_clips"]]
    random.Random(cfg["split_seed"]).shuffle(rows)
    n_val = round(len(rows) * cfg["val_fraction"])
    val, train = rows[:n_val], rows[n_val:]
    out = Path(cfg["dim_mean_path"]).parent / "umt_ssv2_fusedtext_held_out_val_clips.json"
    json.dump([i for i, _ in val], open(out, "w"))
    print(f"  split: {len(train):,} train / {len(val):,} val  (held-out list -> {out.name})")
    return train, val


def build_loader(rows, cfg, shuffle):
    ds = PerturbedKineticsDataset(
        [f"{cfg['video_dir']}/{i}.webm" for i, _ in rows], [i for i, _ in rows],
        [c for _, c in rows], UMTProcessor(), MODEL_REGISTRY["umt"]["num_frames"],
        get_frame_sampler("ssv2", MODEL_REGISTRY["umt"]), "R")
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["num_workers"], pin_memory=True, drop_last=shuffle)


def derive_paths(cfg):
    """Run name, dim_mean path, checkpoints — called once in main, stored back into cfg."""
    sae_dir = ROOT / "outputs" / "sae"
    cfg["run_name"] = (f"sae_umt_ssv2_fusedtext_k{cfg['sae_k']}_x{cfg['expansion']}"
                       f"_l{cfg['layer']}_job{cfg['job_label']}")
    cfg["dim_mean_path"] = sae_dir / f"umt_ssv2_fusedtext_layer{cfg['layer']}_dim_mean.pt"
    cfg["checkpoint"] = sae_dir / f"{cfg['run_name']}.pt"
    cfg["best_checkpoint"] = sae_dir / f"{cfg['run_name']}_best.pt"


@torch.no_grad()
def fused_states(vis, text, w, upto):
    """All 174 templates' hidden states at layer `upto` (9..11) for one clip's vision
    tokens (2352, 768). Layer input is the cache (8). Chunked like vtm_scores."""
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention")
             for i in (9, 10, 11) if i <= upto}
    outs = []
    for s in range(0, len(text["hidden"]), 58):
        h, m = text["hidden"][s:s + 58], text["add_mask"][s:s + 58]
        for i in (9, 10, 11):
            if i > upto:
                break
            h, _ = _fusion_layer_attn(h, m, cross[i], w, i)
        outs.append(h)
    return torch.cat(outs)                      # (174, 25, 768) fp32


@torch.no_grad()
def itm_from(h, start, vis, text, w):
    """Run fusion layers start..11 on h (h is layer start-1's output; may be
    SAE-spliced), then itm_head -> (174,) raw VTM scores."""
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention")
             for i in (9, 10, 11) if i >= start}
    outs = []
    for s in range(0, len(h), 58):
        hh, m = h[s:s + 58], text["add_mask"][s:s + 58]
        for i in (9, 10, 11):
            if i >= start:
                hh, _ = _fusion_layer_attn(hh, m, cross[i], w, i)
        outs.append(_lin(hh[:, 0], w, "itm_head")[:, 1])
    return torch.cat(outs)


def real_tokens(h, real):
    """(174, 25, 768) -> (n_real, 768) with padding positions dropped."""
    return h[real.bool()]


def load_model_and_text(cfg):
    """Vision tower + fusion weights + cached templates; load_vtm runs the xbert
    self-check, real is the (174, 25) padding mask of the cache."""
    from huggingface_hub import hf_hub_download
    model = UMTClassifier.from_pretrained().to(cfg["device"]).eval()
    sd = torch.load(hf_hub_download(UMT_CFG["hf_repo"], UMT_CFG["hf_filename"]),
                    map_location="cpu", weights_only=False)
    text, w = load_vtm(sd, cfg["device"])
    from stage3_analysis.umt_vtm_gate import CFG as GATE_CFG
    real = torch.from_numpy(np.load(GATE_CFG["text_cache"])["mask"]).float().to(cfg["device"])
    return model, text, w, real


@torch.no_grad()
def run_dim_mean(cfg):
    """Per-dim mean of real-token fused states at SAE_LAYER; save fp32 (768,)."""
    train, _ = build_split(cfg)
    loader = build_loader(train[: cfg["dim_mean_clips"]], cfg, shuffle=False)
    model, text, w, real = load_model_and_text(cfg)
    acc, n = torch.zeros(768, device=cfg["device"]), 0
    t0 = time.time()
    for pixel_values, _ in loader:
        x_vis, _ = vision_and_vtc(model, pixel_values.to(cfg["device"]))
        for vis in x_vis:
            t = real_tokens(fused_states(vis, text, w, cfg["layer"]), real)
            acc += t.sum(0).float()
            n += t.shape[0]
        print(f"  {n:,} tokens  {time.time() - t0:.0f}s", flush=True)
    mean = (acc / n).cpu()
    torch.save(mean, cfg["dim_mean_path"])
    print(f"dim_mean ({n:,} tokens) -> {cfg['dim_mean_path']}  abs-mean {mean.abs().mean():.4f}")


def train_epoch(sae, loader, model, text, w, real, dim_mean, optimizer, cfg):
    sae.train()
    total, n_batches = 0.0, 0
    for pixel_values, _ in loader:
        x_vis, _ = vision_and_vtc(model, pixel_values.to(cfg["device"]))
        optimizer.zero_grad()
        loss_sum = 0.0
        for vis in x_vis:
            tokens = (real_tokens(fused_states(vis, text, w, cfg["layer"]), real)
                      - dim_mean).float()
            pre, codes, x_hat = sae(tokens)
            loss = top_k_auxiliary_loss(tokens, x_hat, pre, codes, sae.get_dictionary(),
                                        penalty=cfg["alpha"])
            (loss / len(x_vis)).backward()
            loss_sum += loss.item()
        optimizer.step()
        total += loss_sum / len(x_vis)
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def validate(sae, loader, model, text, w, real, dim_mean, cfg):
    sae.eval()
    ss_res = s_x = ss_x = l0_sum = 0.0
    n_tok = n_clips = 0
    counts = torch.zeros(cfg["nb_concepts"], device=cfg["device"])
    for pixel_values, _ in loader:
        x_vis, _ = vision_and_vtc(model, pixel_values.to(cfg["device"]))
        for vis in x_vis:
            tokens = (real_tokens(fused_states(vis, text, w, cfg["layer"]), real)
                      - dim_mean).float()
            _, codes, x_hat = sae(tokens)
            ss_res += (tokens - x_hat).pow(2).sum().item()
            s_x += tokens.sum().item()
            ss_x += tokens.pow(2).sum().item()
            l0_sum += (codes > 0).float().sum(-1).mean().item()
            counts += (codes > 0).float().sum(0)
            n_tok += tokens.shape[0]
            n_clips += 1
    n_el = n_tok * 768
    mean_x, var_x = s_x / n_el, ss_x / n_el - (s_x / n_el) ** 2
    mse = ss_res / n_el
    return {"r2": 1 - mse / (var_x + 1e-8), "mse": mse, "l0": l0_sum / max(n_clips, 1),
            "dead_features": int((counts == 0).sum().item()), "counts": counts.cpu()}


@torch.no_grad()
def splice_scores(vis, text, w, real, sae, dim_mean, cfg):
    """One clip -> (raw, spliced) VTM scores (174,). Splice: layer-L real tokens ->
    SAE round-trip -> back into the state grid -> fusion resumes at L+1 -> itm."""
    h = fused_states(vis, text, w, cfg["layer"])
    tokens = real_tokens(h, real) - dim_mean
    _, _, x_hat = sae(tokens.float())
    h_spliced = h.clone()
    h_spliced[real.bool()] = x_hat + dim_mean
    return (itm_from(text["hidden"], 9, vis, text, w),
            itm_from(h_spliced, cfg["layer"] + 1, vis, text, w))


@torch.no_grad()
def splice_gate(sae, model, text, w, real, dim_mean, cfg, cap=None):
    """VTM 2-way pair accuracy (score(true) > score(partner)), raw vs spliced, over
    capped val-split clips. Returns (summary_df, per_pair_df)."""
    _, val = build_split(cfg)
    classes = [c for _, c in val[: (cap or cfg["splice_cap"])]]
    loader = build_loader(val[: len(classes)], cfg, shuffle=False)
    raw_all, spl_all = [], []
    for pixel_values, _ in loader:
        x_vis, _ = vision_and_vtc(model, pixel_values.to(cfg["device"]))
        for vis in x_vis:
            r, s = splice_scores(vis, text, w, real, sae, dim_mean, cfg)
            raw_all.append(r.cpu())
            spl_all.append(s.cpu())
    return pairs_2way_fused(torch.stack(raw_all), torch.stack(spl_all), classes)


def pairs_2way_fused(raw, spliced, classes):
    """2-way accuracy over the gate's pair sets; one row per pair + an overall row."""
    rows = []
    for a, b, name, pset in test2_pairs():
        idx = [k for k, y in enumerate(classes) if y in (a, b)]
        if not idx:
            continue
        true = torch.tensor([classes[k] for k in idx])
        partner = torch.where(true == a, b, a)
        m_r = raw[idx, true] - raw[idx, partner]
        m_s = spliced[idx, true] - spliced[idx, partner]
        rows.append(dict(pair=name, set=pset, n=len(idx),
                         raw_2way=float((m_r > 0).float().mean()),
                         spliced_2way=float((m_s > 0).float().mean()),
                         raw_margin_med=float(m_r.median()), spliced_margin_med=float(m_s.median())))
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:      # e.g. smoke: capped val clips cover no pair classes
        return df, {c: float("nan") for c in
                    ["raw_2way", "spliced_2way", "raw_margin_med", "spliced_margin_med"]}
    overall = df[["raw_2way", "spliced_2way", "raw_margin_med", "spliced_margin_med"]].mean()
    return df, overall.to_dict()


@torch.no_grad()
def verb_signature_gate(sae, model, text, w, real, dim_mean, cfg):
    """Null videos (white/black/noise): per-template codes (mean over real tokens)
    at SAE_LAYER; per verb pair, delta across pair members; consistency = mean over
    the top-20 |median delta| features of the fraction of nulls agreeing with the
    median sign. High consistency = the text-injected verb signature is a stable
    SAE-latent property, i.e. set-B extraction is viable."""
    deltas = {name: [] for _, _, name in GEOM_CFG["pairs"]}
    for content, seed in synthetic_videos():
        pv = UMTProcessor()(make_frames(content, seed),
                            return_tensors="pt")["pixel_values"].to(cfg["device"])
        vis = vision_and_vtc(model, pv)[0][0]
        h = fused_states(vis, text, w, cfg["layer"])
        codes = sae((h - dim_mean).float().view(-1, 768))[1].view(h.shape[0], h.shape[1], -1)
        per_tpl = (codes * real[:, :, None]).sum(1) / real.sum(1)[:, None]
        for a, b, name in GEOM_CFG["pairs"]:
            deltas[name].append((per_tpl[a] - per_tpl[b]).cpu())
    return verb_consistency_rows(deltas)


def verb_consistency_rows(deltas, top=20):
    import pandas as pd
    rows = []
    for name, stack in deltas.items():
        D = torch.stack(stack)                       # (n_nulls, C)
        med = D.median(0).values
        sel = med.abs().topk(top).indices
        cons = (D[:, sel].sign() == med[sel].sign()).float().mean(0)
        rows.append(dict(pair=name, top_abs_delta=float(med[sel].abs().mean()),
                         consistency_mean=float(cons.mean()),
                         consistency_min=float(cons.min())))
    return pd.DataFrame(rows)


def setup_sae(cfg, top_k):
    sae = BatchTopKSAE(input_shape=768, nb_concepts=cfg["nb_concepts"],
                       top_k=top_k, device=cfg["device"])
    sae.train()
    return sae, Adam(sae.parameters(), lr=cfg["lr"])


def run_train(cfg):
    import wandb
    dim_mean = torch.load(cfg["dim_mean_path"], weights_only=True).to(cfg["device"])
    train, val = build_split(cfg)
    train_loader = build_loader(train, cfg, shuffle=True)
    val_loader = build_loader(val, cfg, shuffle=False)
    model, text, w, real = load_model_and_text(cfg)
    cfg["nb_concepts"] = cfg["expansion"] * 768
    top_k = cfg["sae_k"] * int(real.sum().item())      # k=64 per real token, per clip call
    sae, optimizer = setup_sae(cfg, top_k)
    print(f"  BatchTopKSAE 768 -> {cfg['nb_concepts']}  top_k={top_k:,} "
          f"({int(real.sum().item()):,} real tokens/clip)  L{cfg['layer']}")
    start_epoch = 0
    if cfg["resume_from"]:
        ckpt = torch.load(cfg["resume_from"], map_location=cfg["device"], weights_only=True)
        sae.load_state_dict(ckpt["sae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        sae.running_threshold = ckpt.get("running_threshold")
        start_epoch = ckpt["epoch"] + 1
        assert start_epoch < cfg["epochs"], "resume checkpoint already complete — stale file?"
        print(f"  resumed from {cfg['resume_from']} at epoch {start_epoch}")
    wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg,
               tags=[f"job_{cfg['job_label']}", "umt_fusedtext"])
    best_score, best_epoch = float("-inf"), None
    if cfg["resume_from"] and cfg["best_checkpoint"].exists():
        best = torch.load(cfg["best_checkpoint"], map_location="cpu", weights_only=True)
        best_score, best_epoch = best.get("score", float("-inf")), best["epoch"] + 1
    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        loss = train_epoch(sae, train_loader, model, text, w, real, dim_mean, optimizer, cfg)
        m = validate(sae, val_loader, model, text, w, real, dim_mean, cfg)
        m.pop("counts")
        score = m["r2"] - 1.5 * m["dead_features"] / cfg["nb_concepts"]
        wandb.log({**{f"val/{k}": v for k, v in m.items()}, "train/loss": loss,
                   "val/score": score, "epoch": epoch + 1})
        print(f"  epoch {epoch + 1}: loss {loss:.4f}  R2 {m['r2']:.4f}  L0 {m['l0']:.1f}  "
              f"dead {m['dead_features']}  score {score:.4f}  ({time.time() - t0:.0f}s)")
        payload = {"epoch": epoch, "sae_state_dict": sae.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "running_threshold": sae.running_threshold, "sae_k": cfg["sae_k"],
                   "score": score, "layer": cfg["layer"]}
        torch.save(payload, cfg["checkpoint"])
        if score > best_score:
            best_score, best_epoch = score, epoch + 1
            torch.save(payload, cfg["best_checkpoint"])
            print(f"  saved epoch {epoch + 1} * new best ({score:.4f})")
    wandb.summary["best_epoch"], wandb.summary["best_score"] = best_epoch, best_score
    wandb.finish()
    try:
        run_eval(cfg, sae=sae, model=model, text=text, w=w, real=real, dim_mean=dim_mean)
    except Exception:
        import traceback
        print("!!! epilogue splice FAILED (non-fatal, checkpoints saved):")
        traceback.print_exc()


def run_eval(cfg, sae=None, model=None, text=None, w=None, real=None, dim_mean=None):
    """Both gates on the best checkpoint: VTM splice + null-video verb signature."""
    if dim_mean is None:
        dim_mean = torch.load(cfg["dim_mean_path"], weights_only=True).to(cfg["device"])
    if sae is None:
        model, text, w, real = load_model_and_text(cfg)
        ckpt = torch.load(cfg["best_checkpoint"], map_location="cpu", weights_only=True)
        cfg["nb_concepts"] = cfg["expansion"] * 768
        sae, _ = setup_sae(cfg, cfg["sae_k"] * int(real.sum().item()))
        sae.load_state_dict(ckpt["sae_state_dict"])
        sae.running_threshold = ckpt.get("running_threshold")
    sae.eval()
    pairs, overall = splice_gate(sae, model, text, w, real, dim_mean, cfg)
    print("\nVTM splice gate (2-way pair accuracy, raw vs SAE-spliced fusion):")
    print(pairs.round(4).to_string(index=False))
    print("overall:", {k: round(v, 4) for k, v in overall.items()})
    print("\nVerb-signature gate (null videos, SAE latents at L%d):" % cfg["layer"])
    print(verb_signature_gate(sae, model, text, w, real, dim_mean, cfg).round(4).to_string(index=False))


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # keep the verified fp32 fusion path
    torch.backends.cudnn.allow_tf32 = False
    derive_paths(CFG)
    ROOT.mkdir(exist_ok=True)
    Path(CFG["dim_mean_path"]).parent.mkdir(parents=True, exist_ok=True)
    print(f"mode={CFG['mode']}  layer={CFG['layer']}  device={CFG['device']}  "
          f"run={CFG['run_name']}  max_clips={CFG['max_clips'] or 'all'}")
    if CFG["mode"] == "dim_mean":
        run_dim_mean(CFG)
    elif CFG["mode"] == "train":
        if not Path(CFG["dim_mean_path"]).exists():
            sys.exit(f"dim_mean missing: {CFG['dim_mean_path']} — run MODE=dim_mean first")
        run_train(CFG)
    elif CFG["mode"] == "eval":
        run_eval(CFG)
    else:
        sys.exit(f"unknown MODE {CFG['mode']!r}")


if __name__ == "__main__":
    main()
