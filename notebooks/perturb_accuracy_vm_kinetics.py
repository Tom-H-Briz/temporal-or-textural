"""
VideoMAE per-class accuracy under conditions R/A/C1 — Kinetics-400 val set.

  R  — real, unperturbed (baseline; no transform)
  A  — single midpoint frame repeated (applied to all native frames, before sampling)
  C1 — shuffled consecutive tubelet pairs (applied to the 16 already-sampled frames)

R is not tubelet-aware and not the model's floor either — it's the actual reference
point A/C1 are both measured against. A destroys temporal info before sampling even
happens (every tubelet ends up an identical repeated-frame pair, so tubelet grouping
is moot for it); C1 preserves each tubelet's internal frame order but scrambles
tubelet order — the two together bracket "no temporal info" against "temporal info
present but mis-ordered", with R as the anchor for both.

K400 uses a different frame sampler than SSv2 (sample_frames_kinetics: a dense,
center-positioned window, not a full-clip linspace) — dispatched via FRAME_SAMPLERS
rather than hardcoded, so this mirrors spliced_accuracy_vm.py's dataset-aware
sampling instead of perturb_accuracy_vm.py's SSv2-only linspace.

Outputs (outputs/stage1_class_selection_VM_kinetics/):
  per_class_accuracy_VM_kinetics_R.csv
  per_class_accuracy_VM_kinetics_A.csv
  per_class_accuracy_VM_kinetics_C1.csv

Usage: uv run python notebooks/perturb_accuracy_vm_kinetics.py
"""

import os
import sys
import zlib
from pathlib import Path

import av
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(Path(__file__).parent))

from perturbationA import apply_midpoint_frame
from perturb_accuracy_vm import apply_shuffle_pairs
from spliced_accuracy_vm import load_kinetics_metadata, per_class_accuracy
from ToT_utils import CHECKPOINT_REGISTRY, DATASET_REGISTRY, FRAME_SAMPLERS, MODEL_REGISTRY

CFG = {
    "model_name":   "videomae",
    "dataset_name": "kinetics400",
    "labels_csv":   os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data/kinetics400/annotations/val.csv")
    ),
    "batch_size":   8,
    "num_workers":  4,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir":   str(ROOT / "outputs/stage1_class_selection_VM_kinetics"),
}

_model_cfg        = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"] = _model_cfg["num_frames"]  # 16
CFG["frame_sampler"] = FRAME_SAMPLERS[CFG["dataset_name"]]  # sample_frames_kinetics


class PerturbedKineticsDataset(Dataset):
    """Condition A operates on all native frames before sampling; C1 operates on
    the num_frames already sampled by frame_sampler. Retries on unreadable clips —
    K400's YouTube-sourced downloads reliably contain some corrupted files at scale
    (same failure mode SSv2ClipDataset guards against in ToT_utils.py)."""

    def __init__(self, clip_paths, labels, processor, num_frames, frame_sampler, condition):
        assert condition in ("R", "A", "C1")
        self.clip_paths = clip_paths
        self.labels = labels
        self.processor = processor
        self.num_frames = num_frames
        self.frame_sampler = frame_sampler
        self.condition = condition

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        frames = None
        for _ in range(5):
            try:
                container = av.open(str(self.clip_paths[idx]))
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
                container.close()
                break
            except Exception as e:
                print(f"  Warning: unreadable clip {self.clip_paths[idx]} ({e}); trying next")
                idx = (idx + 1) % len(self.clip_paths)
        if frames is None:
            raise RuntimeError(f"5 consecutive unreadable clips starting near idx {idx}")

        if self.condition == "A":
            frames = apply_midpoint_frame(frames)  # pre-sampling, on all native frames

        n = len(frames)
        indices = self.frame_sampler(n, self.num_frames)
        sampled = [frames[i] for i in indices]

        if self.condition == "C1":
            # seed from the clip's own filename (stable across runs, unlike Python's
            # randomized str hash()) — K400 clip ids are youtube strings, not ints.
            seed = zlib.crc32(self.clip_paths[idx].stem.encode())
            sampled = apply_shuffle_pairs(sampled, seed)

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]


def run_condition(model, clip_paths, labels, processor, cfg, condition) -> list[int]:
    dataset = PerturbedKineticsDataset(
        clip_paths, labels, processor, cfg["num_frames"], cfg["frame_sampler"], condition
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        num_workers=cfg["num_workers"], pin_memory=True)
    preds = []
    with torch.no_grad():
        for pixel_values, _ in tqdm(loader, desc=f"Condition {condition}"):
            preds.extend(model(pixel_values=pixel_values.to(cfg["device"]))
                         .logits.argmax(dim=-1).cpu().tolist())
    return preds


def save_csv(preds: list[int], labels: list[int], id2label: dict, out_path: Path) -> None:
    # dict-based over whatever class ids actually appear — not range(174) like the
    # SSv2 script, since K400 has 400 classes and a val-clip subset may not hit all.
    acc = per_class_accuracy(preds, labels, id2label)
    rows = [{"class_id": cid, **v} for cid, v in acc.items()]
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(out_path, index=False)
    overall = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"  Overall top-1: {overall:.4f}  -> {out_path.name}")


def main() -> None:
    device = CFG["device"]
    print(f"Device: {device}  Model: {CFG['model_name']}  Dataset: {CFG['dataset_name']}")

    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], CFG["dataset_name"])]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()

    # VIDEO_DIR override for Isambard /scratch mounts — same convention as
    # spliced_accuracy_vm.py; unset here follows train_sae_vm_kinetics.sh's lead
    # and falls back to DATASET_REGISTRY["kinetics400"]["video_dir"].
    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[CFG["dataset_name"]]["video_dir"])
    clip_paths, labels, id2label = load_kinetics_metadata(CFG["labels_csv"], video_dir, model.config.label2id)

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("R", "A", "C1"):
        preds = run_condition(model, clip_paths, labels, processor, CFG, condition)
        save_csv(preds, labels, id2label, out_dir / f"per_class_accuracy_VM_kinetics_{condition}.csv")


if __name__ == "__main__":
    main()
