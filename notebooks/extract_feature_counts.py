"""
Run validation loop against a saved SAE checkpoint and save feature_counts to disk.

Usage on Isambard: submitted via slurm/extract_feature_counts.sh
Output: outputs/sae/feature_counts_job{label}.pt  — scp down for local plotting
"""

import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sae import BatchTopKSAE
from class_selection import load_metadata
from ToT_utils import SSv2ClipDataset

CFG = {
    "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data" / "ssv2_val_set")),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_frames": 16,
    "layer": 7,
    "input_dim": 768,
    "nb_concepts": 768 * 8,
    "k": int(os.environ.get("SAE_K", 64)),
    "job_label": os.environ.get("SAE_JOB_LABEL", "64"),
    "train_clips": 20_000,
    "split_seed": 42,
    "batch_size": 64,
    "num_workers": 4,
    "output_dir": str(ROOT / "outputs" / "sae"),
    "dim_mean_path": str(ROOT / "outputs" / "sae" / "layer7_dim_mean.pt"),
}


def main() -> None:
    device = CFG["device"]
    label = CFG["job_label"]
    checkpoint = Path(CFG["output_dir"]) / f"sae_layer7_job{label}.pt"
    out_path = Path(CFG["output_dir"]) / f"feature_counts_job{label}.pt"

    print(f"Job label: {label}  |  checkpoint: {checkpoint}")

    dim_mean = torch.load(CFG["dim_mean_path"], weights_only=True).to(device)

    print("Loading VideoMAE...")
    processor = VideoMAEImageProcessor.from_pretrained(CFG["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(CFG["model_id"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_storage = {}
    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_storage["activations"] = hidden

    model.videomae.encoder.layer[CFG["layer"]].register_forward_hook(_hook)

    print("Loading SAE checkpoint...")
    sae = BatchTopKSAE(
        input_shape=CFG["input_dim"],
        nb_concepts=CFG["nb_concepts"],
        top_k=CFG["k"] * 1568,
        device=device,
    )
    sae.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    sae.eval()

    print("Building val split...")
    _, clips, _ = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])
    all_paths = [video_dir / f"{c['id']}.webm" for c in clips]
    all_paths = [p for p in all_paths if p.exists()]
    rng = random.Random(CFG["split_seed"])
    rng.shuffle(all_paths)
    val_paths = all_paths[CFG["train_clips"] : CFG["train_clips"] + 4_000]
    print(f"  Val clips: {len(val_paths)}")

    val_ds = SSv2ClipDataset(val_paths, processor, CFG["num_frames"])
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"], pin_memory=True,
    )

    # running_threshold is not persisted in the state dict — warm up with one batch
    sae.train()
    with torch.no_grad():
        warmup = next(iter(val_loader)).to(device)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            model(pixel_values=warmup)
        warmup_acts = hook_storage["activations"].detach() - dim_mean
        sae.encode(warmup_acts[0].float())
    sae.eval()

    feature_counts = torch.zeros(CFG["nb_concepts"], device=device)

    print("Running validation pass...")
    with torch.no_grad():
        for i, pixel_values in enumerate(val_loader):
            pixel_values = pixel_values.to(device)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                model(pixel_values=pixel_values)
            activations = hook_storage["activations"].detach() - dim_mean
            for j in range(activations.shape[0]):
                codes = sae.encode(activations[j].float())[1]
                feature_counts += (codes > 0).float().sum(0)
            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(val_loader)}")

    torch.save(feature_counts.cpu(), out_path)
    print(f"\nSaved → {out_path}")
    print(f"Dead features: {int((feature_counts == 0).sum())}/{CFG['nb_concepts']}")


if __name__ == "__main__":
    main()
