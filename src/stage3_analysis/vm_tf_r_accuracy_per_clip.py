"""
Per-clip R-condition correctness, VM + TF, SL-pilot 35-class population.

Extraction for a paired bootstrap: vm_tf_accuracy_vs_l5_ablation.py's earlier
version resampled VM accuracy, TF accuracy, and L5-ablated accuracy as three
INDEPENDENT binomials — wrong, since VM/TF are scored on the same clips per
class (confirmed: n_vm == n_tf for every class in that table) and some clips
are just easier than others across both backbones. This persists per-clip
(clip_id, class_id, vm_correct, tf_correct) so a real clip-level resample is
possible — mirrors perturb_accuracy_vm_ssv2.py / perturb_accuracy_tf.py's R-
condition loading exactly, but restricted to the 35-class population and
saving per-clip results instead of only the aggregated CSV.

Outputs (outputs/analysis/scaffold_ablation/):
    vm_tf_r_accuracy_per_clip.parquet — clip_id, class_id, vm_correct, tf_correct

Usage:
    uv run python src/stage3_analysis/vm_tf_r_accuracy_per_clip.py
"""

import os
import sys
from functools import partial
from pathlib import Path

import av
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, _strip_brackets, load_metadata, sample_frames_ssv2

OUT_DIR = ROOT / "outputs" / "analysis" / "scaffold_ablation"
CLASS_SOURCE = OUT_DIR / "vm_tf_accuracy_vs_l5_ablation.csv"
LABELS_PATH = os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json"))
VALIDATION_PATH = os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json"))
VIDEO_DIR = os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2"))
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 4


def load_clips() -> tuple[list[Path], list[str], list[int]]:
    """Same load_metadata + template/file-existence filtering as
    perturb_accuracy_vm_ssv2.py / perturb_accuracy_tf.py, restricted to the
    35 classes already in vm_tf_accuracy_vs_l5_ablation.csv — guarantees VM
    and TF are scored on the identical clip set by construction, rather than
    relying on two separate historical scripts having filtered identically."""
    classes = set(pd.read_csv(CLASS_SOURCE)["class_id"])
    label_map, clips, _ = load_metadata(LABELS_PATH, VALIDATION_PATH)
    video_dir = Path(VIDEO_DIR)
    paths, clip_ids, class_ids = [], [], []
    for c in clips:
        template = _strip_brackets(c["template"])
        class_id = label_map.get(template)
        if class_id is None or class_id not in classes:
            continue
        path = video_dir / f"{c['id']}.webm"
        if not path.exists():
            continue
        paths.append(path)
        clip_ids.append(str(c["id"]))
        class_ids.append(class_id)
    print(f"  {len(paths):,} clips across {len(classes)} classes")
    return paths, clip_ids, class_ids


class RDataset(Dataset):
    """R condition only — pure pass-through, no perturbation. frame_sampler
    differs per backbone (VM: sample_frames_ssv2 indices; TF: linspace)."""

    def __init__(self, clip_paths, processor, num_frames, frame_sampler):
        self.clip_paths, self.processor = clip_paths, processor
        self.num_frames, self.frame_sampler = num_frames, frame_sampler

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        container = av.open(str(self.clip_paths[idx]))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        indices = self.frame_sampler(len(frames))
        sampled = [frames[i] for i in indices]
        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values


def _tf_frame_sampler(n: int, num_frames: int) -> list[int]:
    return torch.linspace(0, n - 1, num_frames).long().tolist()


def run_backbone(model_flag: str, clip_paths: list[Path]) -> list[int]:
    model_cfg = MODEL_REGISTRY[model_flag]
    checkpoint = CHECKPOINT_REGISTRY[(model_flag, "ssv2")]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(DEVICE).eval()
    num_frames = model_cfg["num_frames"]

    if model_flag == "videomae":
        frame_sampler = partial(sample_frames_ssv2, num_frames=num_frames)
    else:
        frame_sampler = partial(_tf_frame_sampler, num_frames=num_frames)

    dataset = RDataset(clip_paths, processor, num_frames, frame_sampler)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
    preds = []
    with torch.no_grad():
        for pixel_values in tqdm(loader, desc=f"{model_flag} R"):
            preds.extend(model(pixel_values=pixel_values.to(DEVICE)).logits.argmax(dim=-1).cpu().tolist())
    del model
    return preds


def main() -> None:
    print(f"Device: {DEVICE}")
    clip_paths, clip_ids, class_ids = load_clips()

    vm_preds = run_backbone("videomae", clip_paths)
    tf_preds = run_backbone("timesformer", clip_paths)

    df = pd.DataFrame({
        "clip_id": clip_ids, "class_id": class_ids,
        "vm_correct": [p == c for p, c in zip(vm_preds, class_ids)],
        "tf_correct": [p == c for p, c in zip(tf_preds, class_ids)],
    })
    out_path = OUT_DIR / "vm_tf_r_accuracy_per_clip.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  {len(df):,} clips -> {out_path}")
    print(f"  VM accuracy: {df['vm_correct'].mean():.4f}  TF accuracy: {df['tf_correct'].mean():.4f}")


if __name__ == "__main__":
    main()
