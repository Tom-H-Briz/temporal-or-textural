"""
Per-class spliced vs baseline accuracy for VideoMAE — single layer at a time, whole val set.

Splice = decode(encode(x - dim_mean)) + dim_mean. Matches DFAEngine splice exactly.
Baseline (no splice) and spliced run on the same full validation set.
Mirrors spliced_accuracy_tf.py's structure (single SAE per run, no perturbation conditions).

Usage:
    uv run python notebooks/spliced_accuracy_vm.py --layer 5 --job-label 64
    uv run python notebooks/spliced_accuracy_vm.py --layer 9 --job-label 64

Output: outputs/spliced_accuracy_vm/spliced_accuracy_l{layer}_job{job_label}.csv
"""

import argparse
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
from ToT_utils import (
    MODEL_REGISTRY, SSv2ClipDataset, _strip_brackets,
    load_metadata, make_sae_splice_hook, run_inference,
)

CFG = {
    "model_name":      "videomae",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",       str(ROOT / "data" / "ssv2_val_set")),
    "sae_k":           64,   # matches train_sae_vm_l5_l9.sh recipe — job64 checkpoints only;
                             # older job128/job128_16x/jobe7k128 files use a different k and
                             # would need this overridden to load correctly through this script
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "output_dir":      str(ROOT / "outputs" / "spliced_accuracy_vm"),
    "sae_dir":         str(ROOT / "outputs" / "sae"),
}

_model_cfg              = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"]       = _model_cfg["num_frames"]
CFG["hidden_dim"]       = _model_cfg["hidden_dim"]
CFG["num_patch_tokens"] = _model_cfg["num_patch_tokens"]
CFG["cls_offset"]       = _model_cfg["cls_offset"]


def load_dim_mean(cfg: dict, layer: int) -> torch.Tensor:
    path = Path(cfg["sae_dir"]) / f"layer{layer}_dim_mean.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"dim_mean not found: {path}\n"
            f"Run compute_dim_mean_vm_sweep.sh for layer {layer} first."
        )
    return torch.load(path, weights_only=True).to(cfg["device"])


def load_sae(cfg: dict, layer: int, job_label: str, dim_mean: torch.Tensor) -> BatchTopKSAE:
    """Load a VM SAE checkpoint, warm up running_threshold, return in eval mode."""
    ckpt_path = Path(cfg["sae_dir"]) / f"sae_layer{layer}_job{job_label}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=cfg["device"], weights_only=True)
    state_dict = ckpt["sae_state_dict"] if isinstance(ckpt, dict) and "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    sae = BatchTopKSAE(input_shape=cfg["hidden_dim"], nb_concepts=nb_concepts,
                       top_k=cfg["sae_k"] * cfg["num_patch_tokens"], device=cfg["device"])
    sae.load_state_dict(state_dict)
    sae.train()
    dummy = torch.zeros(cfg["num_patch_tokens"], cfg["hidden_dim"], device=cfg["device"])
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval()
    print(f"  Loaded {ckpt_path.name}  (nb_concepts={nb_concepts})")
    return sae


def per_class_accuracy(preds: list[int], labels: list[int], id2template: dict) -> dict[int, dict]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--job-label", type=str, required=True)
    args = parser.parse_args()

    device  = CFG["device"]
    layer   = args.layer
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Layer: {layer}  Job: {args.job_label}")

    label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])
    all_clips = sorted(
        [c for c in clips if (video_dir / f"{c['id']}.webm").exists()],
        key=lambda c: c["id"],
    )
    paths  = [video_dir / f"{c['id']}.webm" for c in all_clips]
    labels = [label_map[_strip_brackets(c["template"])] for c in all_clips]
    print(f"  {len(all_clips):,} clips (sorted by video ID)")

    dim_mean  = load_dim_mean(CFG, layer)
    model_cfg = MODEL_REGISTRY[CFG["model_name"]]
    processor = model_cfg["processor_class"].from_pretrained(model_cfg["checkpoint"])
    model     = model_cfg["model_class"].from_pretrained(model_cfg["checkpoint"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dataset = SSv2ClipDataset(paths, processor, CFG["num_frames"], labels=labels)
    loader  = DataLoader(dataset, batch_size=CFG["batch_size"],
                         num_workers=CFG["num_workers"], pin_memory=True, shuffle=False)

    print("Running baseline (no splice)...")
    baseline_preds, _ = run_inference(model, loader, device)

    print(f"Loading SAE for layer {layer}...")
    sae    = load_sae(CFG, layer, args.job_label, dim_mean)
    hook_h = model_cfg["layer_getter"](model, layer).register_forward_hook(
        make_sae_splice_hook(sae, dim_mean, CFG["cls_offset"])
    )

    print("Running spliced inference...")
    spliced_preds, _ = run_inference(model, loader, device)
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

    ckpt_name = f"sae_layer{layer}_job{args.job_label}.pt"
    out_path  = out_dir / f"spliced_accuracy_l{layer}_job{args.job_label}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "template", "baseline_accuracy", "spliced_accuracy",
                    "delta", "total_clips", "layer", "checkpoint"])
        for cid in sorted(baseline):
            b = baseline[cid]["accuracy"]
            s = spliced[cid]["accuracy"]
            w.writerow([cid, baseline[cid]["template"], f"{b:.6f}", f"{s:.6f}",
                        f"{s - b:.6f}", baseline[cid]["total"], layer, ckpt_name])
        w.writerow([-1, "OVERALL_CLIP_WEIGHTED", f"{b_overall:.6f}", f"{s_overall:.6f}",
                    f"{s_overall - b_overall:.6f}", n, layer, ckpt_name])
        w.writerow([-1, "OVERALL_MEAN_PER_CLASS", f"{b_mean_pc:.6f}", f"{s_mean_pc:.6f}",
                    f"{s_mean_pc - b_mean_pc:.6f}", len(baseline), layer, ckpt_name])
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
