"""
TimeSformer per-class accuracy under conditions R/C/A — Kinetics-400 val set.

  R  — real, unperturbed (baseline; no transform)
  C  — shuffled frames: ALL native frames permuted pre-sampling (TF convention,
       per perturb_accuracy_tf.py / dfa_mass_delta.py's preprocess_c — raw
       shuffle is in-grammar for TF's single-frame tokens; no C1 pairing)
  A  — single midpoint frame repeated (applied to all native frames, before sampling)

K400 sampler is model-aware: 8 frames over the 64-frame centre window (same
window as VM-K400), dispatched via get_frame_sampler — never FRAME_SAMPLERS
directly. Seeds use _deterministic_seed (crc32 fallback) since K400 clip ids
are YouTube strings.

Outputs (outputs/stage1_class_selection_TF_kinetics/):
  per_class_accuracy_TF_kinetics_R.csv / _C.csv / _A.csv
  comparison.csv   (all three, wide format, R as baseline)

Usage: uv run python notebooks/perturb_accuracy_tf_kinetics.py
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

from perturbation import apply_shuffle
from perturbationA import apply_midpoint_frame
from ToT_utils import (
    CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY, _deterministic_seed,
    get_frame_sampler,
)
from perturb_accuracy_vm_kinetics import save_csv
from spliced_accuracy_vm import load_kinetics_metadata

CFG = {
    "model_name":   "timesformer",
    "dataset_name": "kinetics400",
    "labels_csv":   os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data" / "kinetics400" / "annotations" / "val.csv")
    ),
    "batch_size":   8,
    "num_workers":  4,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir":   str(ROOT / "outputs" / "stage1_class_selection_TF_kinetics"),
}

_model_cfg            = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"]     = _model_cfg["num_frames"]  # 8
CFG["frame_sampler"]  = get_frame_sampler(CFG["dataset_name"], _model_cfg)  # 8x8=64-frame window


class PerturbedKineticsDataset(Dataset):
    """Condition C permutes ALL native frames pre-sampling; A replaces all native
    frames pre-sampling. Retries on unreadable clips — K400's YouTube-sourced
    downloads contain corrupted files at scale (same guard as SSv2ClipDataset)."""

    def __init__(self, clip_paths, clip_ids, labels, processor, num_frames, frame_sampler, condition):
        assert condition in ("R", "C", "A")
        self.clip_paths = clip_paths
        self.clip_ids = clip_ids
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

        if self.condition == "C":
            frames = apply_shuffle(frames, _deterministic_seed(self.clip_ids[idx]))
        elif self.condition == "A":
            frames = apply_midpoint_frame(frames)  # pre-sampling, on all native frames

        n = len(frames)
        indices = self.frame_sampler(n, self.num_frames)
        sampled = [frames[i] for i in indices]
        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]


def run_condition(model, clip_paths, clip_ids, labels, processor, cfg, condition) -> list[int]:
    dataset = PerturbedKineticsDataset(
        clip_paths, clip_ids, labels, processor, cfg["num_frames"], cfg["frame_sampler"], condition
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        num_workers=cfg["num_workers"], pin_memory=True)
    preds = []
    with torch.no_grad():
        for pixel_values, _ in tqdm(loader, desc=f"Condition {condition}"):
            preds.extend(model(pixel_values=pixel_values.to(cfg["device"]))
                         .logits.argmax(dim=-1).cpu().tolist())
    return preds


def merge_conditions(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide comparison, R as baseline — VM version's merge hardcodes C1; TF's set is R/C/A."""
    merged = dfs["R"][["class_id", "template", "total", "correct", "accuracy"]].rename(
        columns={"total": "total_R", "correct": "correct_R", "accuracy": "accuracy_R"}
    )
    for cond in ("C", "A"):
        cols = dfs[cond][["class_id", "correct", "total", "accuracy"]].rename(
            columns={"correct": f"correct_{cond}", "total": f"total_{cond}", "accuracy": f"accuracy_{cond}"}
        )
        merged = merged.merge(cols, on="class_id", how="left")
        merged[f"delta_{cond}_minus_R"] = merged[f"accuracy_{cond}"] - merged["accuracy_R"]
    return merged.sort_values("accuracy_R", ascending=False).reset_index(drop=True)


def main() -> None:
    device = CFG["device"]
    print(f"Device: {device}  Model: {CFG['model_name']}  Dataset: {CFG['dataset_name']}")
    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], CFG["dataset_name"])]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()
    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[CFG["dataset_name"]]["video_dir"])
    clip_paths, labels, id2label = load_kinetics_metadata(CFG["labels_csv"], video_dir, model.config.label2id)
    # clip_ids (filename stems) carried through for the per-clip shuffle seed.
    clip_ids = [p.stem for p in clip_paths]

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs: dict[str, pd.DataFrame] = {}
    for condition in ("R", "C", "A"):
        preds = run_condition(model, clip_paths, clip_ids, labels, processor, CFG, condition)
        dfs[condition] = save_csv(
            preds, labels, id2label, out_dir / f"per_class_accuracy_TF_kinetics_{condition}.csv"
        )
    comp_path = out_dir / "comparison.csv"
    merge_conditions(dfs).to_csv(comp_path, index=False)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
