"""
VideoMAE per-class accuracy under perturbation conditions A/B/C/C1/C2 — full val set.

  A  — single midpoint frame repeated
  B  — first/last frames
  C  — shuffled frames  (seed = int(clip_id) % 2**32)
  C1 — shuffled consecutive pairs of the 16 sampled frames
  C2 — shuffled duplicated-frame pairs (stills) from the 16 sampled frames

Outputs (outputs/stage1_class_selection_VM/):
  per_class_accuracy_VM_A.csv
  per_class_accuracy_VM_B.csv
  per_class_accuracy_VM_C.csv
  per_class_accuracy_VM_C1.csv
  per_class_accuracy_VM_C2.csv

Usage: uv run python notebooks/perturb_accuracy_vm.py
"""

import os
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(Path(__file__).parent))

from perturbation import apply_first_last, apply_shuffle
from perturbationA import apply_midpoint_frame
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, _strip_brackets, load_metadata

CFG = {
    "model_name":      "videomae",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir":      str(ROOT / "outputs/stage1_class_selection_VM"),
}

_model_cfg        = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"] = _model_cfg["num_frames"]  # 16


def apply_shuffle_pairs(frames: list, seed: int) -> list:
    """Shuffle consecutive pairs of 16 sampled frames."""
    pairs = [(frames[i], frames[i + 1]) for i in range(0, len(frames), 2)]
    order = np.random.default_rng(seed).permutation(len(pairs)).tolist()
    result = []
    for i in order:
        result.extend(pairs[i])
    return result


def apply_shuffle_still_pairs(frames: list, seed: int) -> list:
    """Take every other frame, duplicate each, shuffle pair order."""
    evens = frames[::2]  # 8 frames
    pairs = [(f, f) for f in evens]
    order = np.random.default_rng(seed).permutation(len(pairs)).tolist()
    result = []
    for i in order:
        result.extend(pairs[i])
    return result


class PerturbedSSv2Dataset(Dataset):
    def __init__(self, clip_paths, clip_ids, labels, processor, num_frames, condition):
        assert condition in ("A", "B", "C", "C1", "C2")
        self.clip_paths = clip_paths
        self.clip_ids   = clip_ids
        self.labels     = labels
        self.processor  = processor
        self.num_frames = num_frames
        self.condition  = condition

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        container = av.open(str(self.clip_paths[idx]))
        frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        seed = int(self.clip_ids[idx]) % 2**32

        # A/B/C operate on all frames before sampling
        if self.condition == "A":
            frames = apply_midpoint_frame(frames)
        elif self.condition == "B":
            frames = apply_first_last(frames)
        elif self.condition == "C":
            frames = apply_shuffle(frames, seed)

        n       = len(frames)
        indices = torch.linspace(0, n - 1, self.num_frames).long().tolist()
        sampled = [frames[i] for i in indices]

        # C1/C2 operate on the 16 sampled frames
        if self.condition == "C1":
            sampled = apply_shuffle_pairs(sampled, seed)
        elif self.condition == "C2":
            sampled = apply_shuffle_still_pairs(sampled, seed)

        pv = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, self.labels[idx]


def run_condition(model, clip_paths, clip_ids, labels, processor, cfg, condition) -> list[int]:
    dataset = PerturbedSSv2Dataset(clip_paths, clip_ids, labels, processor,
                                   cfg["num_frames"], condition)
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"],
                         num_workers=cfg["num_workers"], pin_memory=True)
    preds   = []
    with torch.no_grad():
        for pixel_values, _ in tqdm(loader, desc=f"Condition {condition}"):
            preds.extend(model(pixel_values=pixel_values.to(cfg["device"]))
                         .logits.argmax(dim=-1).cpu().tolist())
    return preds


def save_csv(preds, labels, id2template, out_path: Path) -> None:
    correct = {i: 0 for i in range(174)}
    total   = {i: 0 for i in range(174)}
    for pred, label in zip(preds, labels):
        total[label]   += 1
        correct[label] += int(pred == label)
    rows = [{"class_id": cid, "template": id2template[cid],
             "correct": correct[cid], "total": total[cid],
             "accuracy": correct[cid] / total[cid] if total[cid] else float("nan")}
            for cid in range(174) if total[cid] > 0]
    pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(out_path, index=False)
    overall = sum(correct.values()) / sum(total.values())
    print(f"  Overall top-1: {overall:.4f}  → {out_path.name}")


def main() -> None:
    device = CFG["device"]
    print(f"Device: {device}  Model: {CFG['model_name']}  Frames: {CFG['num_frames']}")

    label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])

    clip_paths, clip_ids, labels = [], [], []
    for c in clips:
        template = _strip_brackets(c["template"])
        if template not in label_map:
            continue
        path = video_dir / f"{c['id']}.webm"
        if not path.exists():
            continue
        clip_paths.append(path)
        clip_ids.append(c["id"])
        labels.append(label_map[template])
    print(f"  {len(clip_paths):,} clips")

    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], "ssv2")]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("A", "B", "C", "C1", "C2"):
        preds = run_condition(model, clip_paths, clip_ids, labels, processor, CFG, condition)
        save_csv(preds, labels, id2template, out_dir / f"per_class_accuracy_VM_{condition}.csv")


if __name__ == "__main__":
    main()
