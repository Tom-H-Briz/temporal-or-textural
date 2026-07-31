"""
VideoMAE per-class accuracy under conditions R/A/C1 — SSv2 val set.

  R  — real, unperturbed (baseline; no transform)
  A  — single midpoint frame repeated (applied to all native frames, before sampling)
  C1 — shuffled consecutive tubelet pairs (applied to the 16 already-sampled frames)

Mirrors perturb_accuracy_vm_kinetics.py exactly (same three conditions, same merged
comparison.csv, reuses its save_csv/merge_conditions directly) but for SSv2 clips —
perturb_accuracy_vm.py, this project's other SSv2 VM script, computes A/B/C/C1/C2 and
deliberately never computes R, which is what's actually needed here.

Outputs (outputs/stage1_class_selection_VM_ssv2/):
  per_class_accuracy_VM_ssv2_R.csv
  per_class_accuracy_VM_ssv2_A.csv
  per_class_accuracy_VM_ssv2_C1.csv
  comparison.csv

Usage: uv run python notebooks/perturb_accuracy_vm_ssv2.py
"""

import os
import sys
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
from perturb_accuracy_vm_kinetics import apply_shuffle_pairs, merge_conditions, save_csv
from ToT_utils import (
    CHECKPOINT_REGISTRY, MODEL_REGISTRY, _strip_brackets, load_metadata, sample_frames_ssv2,
)

CFG = {
    "model_name":      "videomae",
    "dataset_name":    "ssv2",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "output_dir":      str(ROOT / "outputs/stage1_class_selection_VM_ssv2"),
}

_model_cfg        = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"] = _model_cfg["num_frames"]  # 16


class PerturbedSSv2Dataset(Dataset):
    """Condition A operates on all native frames before sampling; C1 operates on
    the 16 already-sampled frames; R is a pure pass-through (no transform). No
    retry-on-corrupt-clip loop — unlike K400's YouTube downloads, SSv2's curated
    webm set doesn't hit that failure mode (see ToT_utils.SSv2ClipDataset)."""

    def __init__(self, clip_paths, clip_ids, labels, processor, num_frames, condition):
        assert condition in ("R", "A", "C1")
        self.clip_paths = clip_paths
        self.clip_ids = clip_ids
        self.labels = labels
        self.processor = processor
        self.num_frames = num_frames
        self.condition = condition

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        container = av.open(str(self.clip_paths[idx]))
        frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()

        if self.condition == "A":
            frames = apply_midpoint_frame(frames)  # pre-sampling, on all native frames

        n = len(frames)
        indices = sample_frames_ssv2(n, self.num_frames)
        sampled = [frames[i] for i in indices]

        if self.condition == "C1":
            seed = int(self.clip_ids[idx]) % 2**32
            sampled = apply_shuffle_pairs(sampled, seed)

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]


def load_ssv2_clips(cfg: dict) -> tuple[list[Path], list[str], list[int], dict[int, str]]:
    label_map, clips, id2template = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    paths, clip_ids, labels = [], [], []
    for c in clips:
        template = _strip_brackets(c["template"])
        if template not in label_map:
            continue
        path = video_dir / f"{c['id']}.webm"
        if not path.exists():
            continue
        paths.append(path)
        clip_ids.append(str(c["id"]))
        labels.append(label_map[template])
    print(f"  {len(paths):,} clips")
    return paths, clip_ids, labels, id2template


def run_condition(model, clip_paths, clip_ids, labels, processor, cfg, condition) -> list[int]:
    dataset = PerturbedSSv2Dataset(
        clip_paths, clip_ids, labels, processor, cfg["num_frames"], condition
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        num_workers=cfg["num_workers"], pin_memory=True)
    preds = []
    with torch.no_grad():
        for pixel_values, _ in tqdm(loader, desc=f"Condition {condition}"):
            preds.extend(model(pixel_values=pixel_values.to(cfg["device"]))
                         .logits.argmax(dim=-1).cpu().tolist())
    return preds


def main() -> None:
    device = CFG["device"]
    print(f"Device: {device}  Model: {CFG['model_name']}  Dataset: {CFG['dataset_name']}")

    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], CFG["dataset_name"])]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()

    clip_paths, clip_ids, labels, id2label = load_ssv2_clips(CFG)

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for condition in ("R", "A", "C1"):
        preds = run_condition(model, clip_paths, clip_ids, labels, processor, CFG, condition)
        dfs[condition] = save_csv(
            preds, labels, id2label, out_dir / f"per_class_accuracy_VM_ssv2_{condition}.csv"
        )

    comp_path = out_dir / "comparison.csv"
    merge_conditions(dfs).to_csv(comp_path, index=False)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
