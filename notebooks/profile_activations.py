"""
Profile VideoMAE layer-7 activation statistics.
Run locally to understand the distribution before SAE training.

Usage: uv run python notebooks/profile_activations.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from class_selection import load_metadata
from ToT_utils import SSv2ClipDataset

CFG = {
    "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
    "labels_path": str(ROOT / "data" / "ssv2" / "labels" / "labels.json"),
    "validation_path": str(ROOT / "data" / "ssv2" / "labels" / "validation.json"),
    "video_dir": str(ROOT / "data" / "ssv2_val_set"),
    "num_frames": 16,
    "layer": 7,
    "n_clips": 2000,
    "batch_size": 8,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def main():
    device = CFG["device"]
    print(f"Device: {device}")

    print("Loading model...")
    processor = VideoMAEImageProcessor.from_pretrained(CFG["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(CFG["model_id"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_storage = {}
    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_storage["activations"] = hidden.detach()

    model.videomae.encoder.layer[CFG["layer"]].register_forward_hook(_hook)

    print("Loading clips...")
    _, clips, _ = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])
    paths = [video_dir / f"{c['id']}.webm" for c in clips]
    paths = [p for p in paths if p.exists()][: CFG["n_clips"]]
    print(f"  Using {len(paths)} clips")

    dataset = SSv2ClipDataset(paths, processor, CFG["num_frames"])
    loader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=False, num_workers=2)

    all_acts = []
    print("Running forward passes...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            model(pixel_values=batch)
            # collect all tokens from all clips in batch: (B, 1568, 768)
            all_acts.append(hook_storage["activations"].float().cpu())

    acts = torch.cat(all_acts, dim=0)  # (N_clips, 1568, 768)
    flat = acts.reshape(-1, acts.shape[-1])  # (N_clips*1568, 768)
    flat_all = acts.reshape(-1)              # all elements

    print(f"\n--- Layer-{CFG['layer']} activation statistics ---")
    print(f"Shape: {acts.shape}  ({acts.shape[0]*acts.shape[1]:,} tokens total)")
    print()
    print(f"Global scalar mean:  {flat_all.mean():.4f}")
    print(f"Global scalar std:   {flat_all.std():.4f}")
    print(f"Global scalar var:   {flat_all.var():.4f}")
    print(f"Global min / max:    {flat_all.min():.3f} / {flat_all.max():.3f}")
    print()

    per_dim_mean = flat.mean(0)  # (768,)
    print(f"Per-dim mean — min: {per_dim_mean.min():.4f},  max: {per_dim_mean.max():.4f},  "
          f"abs-mean: {per_dim_mean.abs().mean():.4f},  std: {per_dim_mean.std():.4f}")

    per_dim_std = flat.std(0)
    print(f"Per-dim std  — min: {per_dim_std.min():.4f},  max: {per_dim_std.max():.4f},  "
          f"mean: {per_dim_std.mean():.4f}")

    ratio = per_dim_mean.abs().mean() / per_dim_std.mean()
    print(f"\nRatio |mean| / std:  {ratio:.4f}  (>0.3 suggests mean-subtraction is worthwhile)")

    # Token-level: how much does the mean vary across token positions?
    per_token_mean = acts.mean(0)   # (1568, 768) — mean over clips
    token_mean_norm = per_token_mean.norm(dim=-1)  # (1568,)
    print(f"\nPer-token mean vector norm — min: {token_mean_norm.min():.3f}, "
          f"max: {token_mean_norm.max():.3f}, mean: {token_mean_norm.mean():.3f}")
    print("(High variance here = token-position-specific means = position bias)")


if __name__ == "__main__":
    main()
