"""
Per-class spliced vs baseline accuracy for VideoMAE — single layer at a time.

Splice = decode(encode(x - dim_mean)) + dim_mean. Matches DFAEngine splice exactly.
Baseline (no splice) and spliced run on the same clip set.
Mirrors spliced_accuracy_tf.py's structure (single SAE per run, no perturbation conditions).

Callable standalone (CLI, explicit checkpoint path required) or imported and called
directly from train_sae.py at the end of training — run_spliced_accuracy() is the
shared entry point either way; nothing here reconstructs a checkpoint path from a
job label, so old- and new-style filenames both work as long as the path is explicit.

Usage:
    uv run python notebooks/spliced_accuracy_vm.py --layer 5 --dataset-name ssv2 \\
        --sae-checkpoint outputs/sae/sae_layer5_job64.pt

Output: outputs/spliced_accuracy_vm/spliced_accuracy_l{layer}_{dataset_name}_{ckpt_stem}.csv
"""

import argparse
import csv
import json
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
    CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY, SSv2ClipDataset,
    _strip_brackets, load_metadata, make_sae_splice_hook, run_inference,
)

CFG = {
    "model_name":      "videomae",
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "output_dir":      str(ROOT / "outputs" / "spliced_accuracy_vm"),
    "sae_dir":         str(ROOT / "outputs" / "sae"),
    "kinetics_labels_csv": os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data" / "kinetics400" / "annotations" / "val.csv")
    ),
}

_model_cfg              = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"]       = _model_cfg["num_frames"]
CFG["hidden_dim"]       = _model_cfg["hidden_dim"]
CFG["num_patch_tokens"] = _model_cfg["num_patch_tokens"]
CFG["cls_offset"]       = _model_cfg["cls_offset"]


def load_dim_mean(cfg: dict, dataset_name: str, layer: int) -> torch.Tensor:
    # Must match train_sae.py's dim_mean naming exactly (model-abbrev_dataset_layerN) —
    # this script only ever runs against videomae, hence the fixed "vmae" abbrev.
    path = Path(cfg["sae_dir"]) / f"vmae_{dataset_name}_layer{layer}_dim_mean.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"dim_mean not found: {path}\n"
            f"Run compute_dim_mean_vm_sweep.sh (or _kinetics_sweep) for layer {layer} first."
        )
    return torch.load(path, weights_only=True).to(cfg["device"])


def load_sae(cfg: dict, sae_checkpoint: Path, dim_mean: torch.Tensor) -> BatchTopKSAE:
    """Load a VM SAE checkpoint from an explicit path, warm up running_threshold, return in eval mode."""
    ckpt_path = Path(sae_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=cfg["device"], weights_only=True)
    state_dict = ckpt["sae_state_dict"] if isinstance(ckpt, dict) and "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    # sae_k read from the checkpoint itself, not a fixed CFG default — this sweep mixes
    # k=64 and k=128 checkpoints, and a wrong top_k silently changes reconstruction quality.
    assert isinstance(ckpt, dict) and "sae_k" in ckpt, f"Checkpoint missing sae_k: {ckpt_path}"
    sae = BatchTopKSAE(input_shape=cfg["hidden_dim"], nb_concepts=nb_concepts,
                       top_k=ckpt["sae_k"] * cfg["num_patch_tokens"], device=cfg["device"])
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


def load_kinetics_metadata(
    labels_csv: str, video_dir: Path, label2id: dict[str, int]
) -> tuple[list[Path], list[int], dict[int, str]]:
    """Parse the standard DeepMind Kinetics annotation CSV (label, youtube_id, time_start,
    time_end, split); match each row to a clip on disk via glob (timestamped filename,
    falling back to bare youtube_id). Label ids come from the finetuned head's own
    label2id, not re-derived from the CSV — so predictions and ground truth share one id space.
    """
    norm_label2id = {k.strip().lower(): v for k, v in label2id.items()}
    paths, labels, unmatched_label, unmatched_file = [], [], 0, 0
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            label_id = norm_label2id.get(row["label"].strip().lower())
            if label_id is None:
                unmatched_label += 1
                continue
            start, end = int(float(row["time_start"])), int(float(row["time_end"]))
            candidates = (list(video_dir.glob(f"{row['youtube_id']}_{start:06d}_{end:06d}.*"))
                          or list(video_dir.glob(f"{row['youtube_id']}.*")))
            if not candidates:
                unmatched_file += 1
                continue
            paths.append(candidates[0])
            labels.append(label_id)
    print(f"  {len(paths):,} K400 clips matched  "
          f"({unmatched_label} unmatched labels, {unmatched_file} missing files)")
    return paths, labels, {v: k for k, v in label2id.items()}


def load_eval_set(
    cfg: dict, dataset_name: str, video_dir: Path, model, eval_clips: list[str] | None,
) -> tuple[list[Path], list[int], dict[int, str]]:
    """Dispatch to SSv2 (labels.json/validation.json) or K400 (annotation CSV) metadata,
    then optionally restrict to an explicit eval_clips filename list — the persisted
    held-out split — so training-val and spliced-accuracy reuse the same clips."""
    if dataset_name == "ssv2":
        d = DATASET_REGISTRY["ssv2"]
        labels_path     = os.environ.get("LABELS_PATH")     or str(d["labels_path"])
        validation_path = os.environ.get("VALIDATION_PATH") or str(d["validation_path"])
        label_map, clips, id2label = load_metadata(labels_path, validation_path)
        all_clips = sorted([c for c in clips if (video_dir / f"{c['id']}.webm").exists()], key=lambda c: c["id"])
        paths  = [video_dir / f"{c['id']}.webm" for c in all_clips]
        labels = [label_map[_strip_brackets(c["template"])] for c in all_clips]
    elif dataset_name == "kinetics400":
        paths, labels, id2label = load_kinetics_metadata(cfg["kinetics_labels_csv"], video_dir, model.config.label2id)
    else:
        raise ValueError(f"No eval-set loader for dataset_name={dataset_name!r}")

    if eval_clips is not None:
        keep = set(eval_clips)
        filtered = [(p, l) for p, l in zip(paths, labels) if p.name in keep]
        assert filtered, f"None of {len(eval_clips)} eval_clips matched clips on disk in {video_dir}"
        paths, labels = (list(x) for x in zip(*filtered))

    print(f"  {len(paths):,} eval clips ({dataset_name})")
    return paths, labels, id2label


def run_spliced_accuracy(
    *, sae_checkpoint: str, layer: int, model_name: str, dataset_name: str,
    eval_clips: list[str] | None = None, cfg: dict = CFG,
) -> dict:
    """Baseline vs spliced clip-weighted accuracy for one SAE checkpoint. Raw per-clip
    results are written to CSV before the summary scalars are computed (two-stage
    extraction discipline), then baseline/spliced/drop are returned for the caller
    (e.g. train_sae.py) to log to WandB as a prominent summary metric."""
    device  = cfg["device"]
    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Layer: {layer}  Dataset: {dataset_name}  Checkpoint: {sae_checkpoint}")

    dim_mean   = load_dim_mean(cfg, dataset_name, layer)
    model_cfg  = MODEL_REGISTRY[model_name]
    checkpoint = CHECKPOINT_REGISTRY[(model_name, dataset_name)]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # VIDEO_DIR overrides the registry default — Isambard runs mount clips under
    # /scratch, not ROOT/data, matching train_sae.py's override pattern.
    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[dataset_name]["video_dir"])
    paths, labels, id2label = load_eval_set(cfg, dataset_name, video_dir, model, eval_clips)

    dataset = SSv2ClipDataset(paths, processor, cfg["num_frames"], labels=labels)
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"],
                         num_workers=cfg["num_workers"], pin_memory=True, shuffle=False)

    print("Running baseline (no splice)...")
    baseline_preds, _ = run_inference(model, loader, device)

    print(f"Loading SAE for layer {layer}...")
    sae    = load_sae(cfg, sae_checkpoint, dim_mean)
    hook_h = model_cfg["layer_getter"](model, layer).register_forward_hook(
        make_sae_splice_hook(sae, dim_mean, cfg["cls_offset"])
    )

    print("Running spliced inference...")
    spliced_preds, _ = run_inference(model, loader, device)
    hook_h.remove()

    baseline = per_class_accuracy(baseline_preds, labels, id2label)
    spliced  = per_class_accuracy(spliced_preds,  labels, id2label)

    n = len(labels)
    b_overall = sum(p == l for p, l in zip(baseline_preds, labels)) / n
    s_overall = sum(p == l for p, l in zip(spliced_preds,  labels)) / n
    print(f"\n  Clip-weighted:  baseline={b_overall:.4f}  spliced={s_overall:.4f}  "
          f"drop={b_overall - s_overall:+.4f}")

    ckpt_stem = Path(sae_checkpoint).stem
    out_path  = out_dir / f"spliced_accuracy_l{layer}_{dataset_name}_{ckpt_stem}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "template", "baseline_accuracy", "spliced_accuracy",
                    "delta", "total_clips", "layer", "checkpoint"])
        for cid in sorted(baseline):
            b, s = baseline[cid]["accuracy"], spliced[cid]["accuracy"]
            w.writerow([cid, baseline[cid]["template"], f"{b:.6f}", f"{s:.6f}",
                        f"{s - b:.6f}", baseline[cid]["total"], layer, ckpt_stem])
        w.writerow([-1, "OVERALL_CLIP_WEIGHTED", f"{b_overall:.6f}", f"{s_overall:.6f}",
                    f"{s_overall - b_overall:.6f}", n, layer, ckpt_stem])
    print(f"Saved raw per-clip results -> {out_path}")

    return {
        "baseline_accuracy_clip_weighted":     b_overall,
        "spliced_accuracy_clip_weighted":      s_overall,
        "spliced_accuracy_drop_clip_weighted": b_overall - s_overall,
        "csv_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--dataset-name", type=str, required=True, choices=list(DATASET_REGISTRY))
    parser.add_argument("--sae-checkpoint", type=str, required=True)
    parser.add_argument("--eval-clips", type=str, default=None,
                         help="Path to a JSON list of clip filenames (e.g. a held-out "
                              "split); omit to use every clip found for the dataset")
    args = parser.parse_args()

    eval_clips = json.load(open(args.eval_clips)) if args.eval_clips else None
    run_spliced_accuracy(
        sae_checkpoint=args.sae_checkpoint, layer=args.layer, model_name=CFG["model_name"],
        dataset_name=args.dataset_name, eval_clips=eval_clips,
    )


if __name__ == "__main__":
    main()
