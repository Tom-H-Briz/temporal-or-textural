"""
ViViT-B/16x2 per-class accuracy under conditions R/A/C1 — Kinetics-400 val set.

  R  — real, unperturbed (baseline; no transform)
  A  — single midpoint frame repeated (applied to all native frames, before sampling)
  C1 — shuffled consecutive tubelet pairs (applied to the 32 already-sampled frames)

ViViT's tokenizer matches VideoMAE's (tubelet conv3d, temporal tubelet size 2) but
ingests 32 frames (config video_size [32,224,224]) -> 16 tubelet positions. C1's
consecutive-frame-pair shuffle therefore still preserves every tubelet's internal
order — same semantics as the VM variant, at double the positions.

google/vivit-b-16x2-kinetics400 ships generic LABEL_N placeholders in its config,
so label2id comes from resolve_k400_label2id (canonical alphabetical 400-class
fallback) instead of model.config.label2id — a wrong mapping shows up here as
~0.25% top-1 on R. The checkpoint's preprocessor config ALSO double-normalizes
under the pinned transformers (offset-rescale -> [-1,1], then normalize 0.5/0.5
-> [-3,+1]; a live upstream bug), so the processor comes from get_processor with
do_normalize=False. Both prior runs (R 0.5687 stride-4, 0.5648 stride-2) are
PRE-FIX artifacts of exactly this — expect R in the high-70s now (paper 80.0 is
4-view; this pipeline is single-view like VM-K400).

K400 uses a different frame sampler than SSv2 (sample_frames_kinetics: a dense,
center-positioned window, not a full-clip linspace) — dispatched via
get_frame_sampler, which applies ViViT's own frame_sample_rate=2 (paper §4.1;
VM's kinetics protocol is rate 4), so the window is 32*2=64 frames and C1's
frame pairs are truly consecutive native frames. Single center view, same as
the VM-K400 protocol in this pipeline (no multi-view logit averaging).

Outputs (outputs/stage1_class_selection_VIVIT_kinetics/):
  per_class_accuracy_VIVIT_kinetics_R.csv
  per_class_accuracy_VIVIT_kinetics_A.csv
  per_class_accuracy_VIVIT_kinetics_C1.csv
  comparison.csv   (all three, wide format, R as baseline — mirrors perturb_accuracy.py)

Usage: uv run python notebooks/perturb_accuracy_vivit_kinetics.py
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
from ToT_utils import (
    CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY,
    get_frame_sampler, get_processor, resolve_k400_label2id,
)

CFG = {
    "model_name":   "vivit",
    "dataset_name": "kinetics400",
    "labels_csv":   os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data/kinetics400/annotations/val.csv")
    ),
    "batch_size":   8,
    "num_workers":  4,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir":   str(ROOT / "outputs/stage1_class_selection_VIVIT_kinetics"),
}

_model_cfg        = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"] = _model_cfg["num_frames"]  # 32
CFG["frame_sampler"] = get_frame_sampler(CFG["dataset_name"], _model_cfg)  # rate 2 -> 64-frame window


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


def save_csv(preds: list[int], labels: list[int], id2label: dict, out_path: Path) -> pd.DataFrame:
    # dict-based over whatever class ids actually appear — not range(174) like the
    # SSv2 script, since K400 has 400 classes and a val-clip subset may not hit all.
    acc = per_class_accuracy(preds, labels, id2label)
    rows = [{"class_id": cid, **v} for cid, v in acc.items()]
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(out_path, index=False)
    overall = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    print(f"  Overall top-1: {overall:.4f}  -> {out_path.name}")
    return df  # returned so main() can build the merged comparison without a disk re-read


def merge_conditions(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide comparison table, R as baseline — mirrors perturb_accuracy.py's comparison.csv."""
    merged = dfs["R"][["class_id", "template", "total", "correct", "accuracy"]].rename(
        columns={"total": "total_R", "correct": "correct_R", "accuracy": "accuracy_R"}
    )
    for cond in ("A", "C1"):
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
    processor  = get_processor(model_cfg, checkpoint)  # do_normalize=False — see docstring
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()

    # VIDEO_DIR override for Isambard /scratch mounts — same convention as
    # spliced_accuracy_vm.py; unset here follows train_sae_vm_kinetics.sh's lead
    # and falls back to DATASET_REGISTRY["kinetics400"]["video_dir"].
    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[CFG["dataset_name"]]["video_dir"])
    # vivit's config carries LABEL_N placeholders — resolve_k400_label2id falls back
    # to the canonical alphabetical list, so predictions and ground truth share one id space
    label2id = resolve_k400_label2id(CFG["model_name"])
    clip_paths, labels, id2label = load_kinetics_metadata(CFG["labels_csv"], video_dir, label2id)

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for condition in ("R", "A", "C1"):
        preds = run_condition(model, clip_paths, labels, processor, CFG, condition)
        dfs[condition] = save_csv(
            preds, labels, id2label, out_dir / f"per_class_accuracy_VIVIT_kinetics_{condition}.csv"
        )

    comp_path = out_dir / "comparison.csv"
    merge_conditions(dfs).to_csv(comp_path, index=False)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
