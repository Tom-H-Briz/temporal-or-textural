"""
Per-class spliced vs baseline accuracy for TimeSformer — single layer at a time.

Splice = decode(encode(x − dim_mean)) + dim_mean. Matches DFAEngine splice exactly.
CLS token (position 0) is split off, patch tokens reconstructed, CLS recombined.
Baseline (no splice) and spliced run on the same pinned clip set.

Usage:
    SAE_LAYER=7 uv run python notebooks/spliced_accuracy_tf.py

Output: outputs/spliced_accuracy_tf/spliced_accuracy_l{layer}.csv
"""

import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from sae import BatchTopKSAE
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, SSv2ClipDataset, _strip_brackets, load_metadata

CFG = {
    "model_name":      "timesformer",
    "layer":           int(os.environ.get("SAE_LAYER", 7)),
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",       str(ROOT / "data" / "ssv2_val_set")),
    "sae_k":           64,
    "sae_expansion":   8,
    "sae_abbrev":      "tf",
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir":      str(ROOT / "outputs" / "spliced_accuracy_tf"),
    "sae_dir":         str(ROOT / "outputs" / "sae"),
}

_model_cfg          = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"]   = _model_cfg["num_frames"]
CFG["hidden_dim"]   = _model_cfg["hidden_dim"]
CFG["num_patch_tokens"] = _model_cfg["num_patch_tokens"]
CFG["nb_concepts"]  = CFG["sae_expansion"] * CFG["hidden_dim"]
CFG["cls_offset"]   = _model_cfg["cls_offset"]


def load_dim_mean(cfg: dict) -> torch.Tensor:
    path = Path(cfg["sae_dir"]) / f"{cfg['sae_abbrev']}_layer{cfg['layer']}_dim_mean.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"dim_mean not found: {path}\n"
            f"Run compute_dim_mean_sweep.sh for layer {cfg['layer']} first."
        )
    return torch.load(path, weights_only=True).to(cfg["device"])


def load_sae(cfg: dict, dim_mean: torch.Tensor) -> BatchTopKSAE:
    """Load _best.pt, warmup running_threshold with zeros (matches DFAEngine), eval."""
    stem = (f"sae_{cfg['sae_abbrev']}_k{cfg['sae_k']}_x{cfg['sae_expansion']}"
            f"_l{cfg['layer']}_job{cfg['layer']}_best.pt")
    ckpt_path = Path(cfg["sae_dir"]) / stem
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")
    ckpt        = torch.load(ckpt_path, map_location=cfg["device"], weights_only=True)
    state_dict  = ckpt["sae_state_dict"]
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    sae = BatchTopKSAE(input_shape=cfg["hidden_dim"], nb_concepts=nb_concepts,
                       top_k=cfg["sae_k"] * cfg["num_patch_tokens"], device=cfg["device"])
    sae.load_state_dict(state_dict)
    sae.train()
    dummy = torch.zeros(cfg["num_patch_tokens"], cfg["hidden_dim"], device=cfg["device"])
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"  Loaded {stem}  (best epoch={epoch})")
    return sae


def make_splice_hook(sae: BatchTopKSAE, dim_mean: torch.Tensor, cls_offset: int):
    """
    Splice hook matching DFAEngine._splice_hook exactly, with CLS handling for TF.
    Patch tokens only are encoded/decoded; CLS (position 0) is passed through unchanged.
    """
    def _hook(module, input, output):
        hidden  = output[0] if isinstance(output, tuple) else output  # (B, 1+T, D)
        cls     = hidden[:, :cls_offset]     # (B, 1, D) — untouched
        patches = hidden[:, cls_offset:]     # (B, T, D)
        B, T, D = patches.shape
        tokens  = (patches.reshape(B * T, D) - dim_mean).float()
        with torch.no_grad():
            _, z = sae.encode(tokens)
            recon = sae.decode(z.detach())
        recon = (recon + dim_mean).to(hidden.dtype).reshape(B, T, D)
        out   = torch.cat([cls, recon], dim=1) if cls_offset else recon
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return _hook


def run_inference(model, loader: DataLoader, device: str) -> list[int]:
    model.eval()
    preds = []
    with torch.no_grad():
        for pixel_values in loader:
            logits = model(pixel_values=pixel_values.to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return preds


def per_class_accuracy(preds: list[int], labels: list[int],
                        id2template: dict) -> dict[int, dict]:
    counts: dict[int, dict] = {}
    for pred, label in zip(preds, labels):
        if label not in counts:
            counts[label] = {"template": id2template[label], "correct": 0, "total": 0}
        counts[label]["total"]   += 1
        counts[label]["correct"] += int(pred == label)
    for v in counts.values():
        v["accuracy"] = v["correct"] / v["total"] if v["total"] else 0.0
    return counts


def main() -> None:
    device    = CFG["device"]
    layer     = CFG["layer"]
    out_dir   = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Layer: {layer}")

    label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])
    all_clips = sorted(
        [c for c in clips if (video_dir / f"{c['id']}.webm").exists()],
        key=lambda c: c["id"],
    )
    paths  = [video_dir / f"{c['id']}.webm" for c in all_clips]
    labels = [label_map[_strip_brackets(c["template"])] for c in all_clips]
    print(f"  {len(all_clips):,} clips (sorted by video ID)")

    dim_mean   = load_dim_mean(CFG)
    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], "ssv2")]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dataset = SSv2ClipDataset(paths, processor, CFG["num_frames"])
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"],
                         num_workers=CFG["num_workers"], pin_memory=True, shuffle=False)

    print("Running baseline (no splice)...")
    baseline_preds = run_inference(model, loader, device)

    print(f"Loading SAE for layer {layer}...")
    sae    = load_sae(CFG, dim_mean)
    hook_h = model_cfg["layer_getter"](model, layer).register_forward_hook(
        make_splice_hook(sae, dim_mean, CFG["cls_offset"])
    )

    print("Running spliced inference...")
    spliced_preds = run_inference(model, loader, device)
    hook_h.remove()

    baseline = per_class_accuracy(baseline_preds, labels, id2template)
    spliced  = per_class_accuracy(spliced_preds,  labels, id2template)

    n = len(labels)
    b_overall = sum(p == l for p, l in zip(baseline_preds, labels)) / n
    s_overall = sum(p == l for p, l in zip(spliced_preds,  labels)) / n
    b_mean_pc = sum(v["accuracy"] for v in baseline.values()) / len(baseline)
    s_mean_pc = sum(v["accuracy"] for v in spliced.values())  / len(spliced)
    print(f"\n  Clip-weighted:  baseline={b_overall:.4f}  spliced={s_overall:.4f}  "
          f"delta={s_overall - b_overall:+.4f}")
    print(f"  Mean per-class: baseline={b_mean_pc:.4f}  spliced={s_mean_pc:.4f}  "
          f"delta={s_mean_pc - b_mean_pc:+.4f}")

    ckpt_stem = (f"sae_{CFG['sae_abbrev']}_k{CFG['sae_k']}_x{CFG['sae_expansion']}"
                 f"_l{layer}_job{layer}_best.pt")
    out_path  = out_dir / f"spliced_accuracy_l{layer}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "template", "baseline_accuracy", "spliced_accuracy",
                    "delta", "total_clips", "layer", "checkpoint"])
        for cid in sorted(baseline):
            b = baseline[cid]["accuracy"]
            s = spliced[cid]["accuracy"]
            w.writerow([cid, baseline[cid]["template"], f"{b:.6f}", f"{s:.6f}",
                        f"{s - b:.6f}", baseline[cid]["total"], layer, ckpt_stem])
        w.writerow([-1, "OVERALL_CLIP_WEIGHTED", f"{b_overall:.6f}", f"{s_overall:.6f}",
                    f"{s_overall - b_overall:.6f}", n, layer, ckpt_stem])
        w.writerow([-1, "OVERALL_MEAN_PER_CLASS", f"{b_mean_pc:.6f}", f"{s_mean_pc:.6f}",
                    f"{s_mean_pc - b_mean_pc:.6f}", len(baseline), layer, ckpt_stem])
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
