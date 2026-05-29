"""
Train BatchTopK SAE on VideoMAE layer-7 residual stream.

SSv2 validation set, real clips only — 20k train / 4k val, random split.
Output: outputs/sae/sae_layer_7.pt  (overwritten each epoch)

REMEMBER!!!! set WANDB_API_KEY env var in job script before running!
"""

import os
import random
import sys
from pathlib import Path

import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sae import BatchTopKSAE
from sae.losses import top_k_auxiliary_loss
from class_selection import load_metadata
from ToT_utils import SSv2ClipDataset

CFG = {
    "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data" / "ssv2_val_set")),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # VideoMAE
    "num_frames": 16,
    "layer": 7,
    # SAE architecture
    "input_dim": 768,
    "nb_concepts": 768 * 8,    # 6144
    "top_k": 64 * 1568,        # 100_352 — one clip's token budget
    "aux_loss_coeff": 0.03,
    # Training
    "epochs": 5,
    "batch_size": 64,           # clips per gradient step
    "lr": 1e-4,
    "train_clips": 20_000,
    "split_seed": 42,
    "num_workers": 4,
    # Output
    "output_dir": str(ROOT / "outputs" / "sae"),
    "checkpoint": str(ROOT / "outputs" / "sae" / "sae_layer_7.pt"),
    "wandb_project": "temporal-or-textural",
    "wandb_run": "sae_layer7_batchtopk",
}


def build_split(cfg: dict) -> tuple[list[Path], list[Path]]:
    _, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])

    all_paths = [video_dir / f"{c['id']}.webm" for c in clips]
    all_paths = [p for p in all_paths if p.exists()]
    print(f"  {len(all_paths):,} clips on disk  ({len(clips):,} in validation.json)")

    rng = random.Random(cfg["split_seed"])
    rng.shuffle(all_paths)

    if len(all_paths) < cfg["train_clips"]:
        print(f"  Warning: fewer clips than train_clips={cfg['train_clips']:,}; using all for train")
        return all_paths, []

    train_paths = all_paths[: cfg["train_clips"]]
    val_paths = all_paths[cfg["train_clips"] : cfg["train_clips"] + 4_000]
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}")
    return train_paths, val_paths


def build_loaders(
    train_paths: list[Path], val_paths: list[Path], processor, cfg: dict
) -> tuple[DataLoader, DataLoader]:
    train_ds = SSv2ClipDataset(train_paths, processor, cfg["num_frames"])
    val_ds = SSv2ClipDataset(val_paths, processor, cfg["num_frames"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


def setup_videomae(cfg: dict) -> tuple:
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_id"])
    model.to(cfg["device"]).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_storage: dict = {}

    def _hook(module, input, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        hook_storage["activations"] = hidden  # (B, 1568, 768)

    model.videomae.encoder.layer[cfg["layer"]].register_forward_hook(_hook)
    return model, processor, hook_storage


def setup_sae(cfg: dict) -> tuple[BatchTopKSAE, Adam]:
    sae = BatchTopKSAE(
        input_shape=cfg["input_dim"],
        nb_concepts=cfg["nb_concepts"],
        top_k=cfg["top_k"],
        device=cfg["device"],
    )
    sae.train()
    optimizer = Adam(sae.parameters(), lr=cfg["lr"])
    return sae, optimizer


def train_epoch(
    sae: BatchTopKSAE,
    loader: DataLoader,
    videomae,
    hook_storage: dict,
    optimizer: Adam,
    cfg: dict,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    sae.train()
    device = cfg["device"]
    total_loss = 0.0

    for pixel_values in loader:
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                videomae(pixel_values=pixel_values)

        # (B, 1568, 768) — detached, float16 on GPU
        activations = hook_storage["activations"].detach()
        n_clips = activations.shape[0]

        optimizer.zero_grad()
        batch_loss = 0.0

        # Per-clip BatchTopK: sparsity budget is allocated within each clip's 1568-token pool
        for i in range(n_clips):
            tokens = activations[i].float()  # (1568, 768) float32
            pre_codes, codes, x_hat = sae(tokens)
            loss = top_k_auxiliary_loss(
                tokens, x_hat, pre_codes, codes,
                sae.get_dictionary(), penalty=cfg["aux_loss_coeff"],
            )
            (loss / n_clips).backward()
            batch_loss += loss.item()

        optimizer.step()

        batch_loss /= n_clips
        total_loss += batch_loss
        wandb.log({"train/loss": batch_loss, "epoch": epoch + 1}, step=global_step)
        global_step += 1

    return total_loss / len(loader), global_step


def validate(
    sae: BatchTopKSAE,
    loader: DataLoader,
    videomae,
    hook_storage: dict,
    cfg: dict,
) -> dict:
    sae.eval()  # triggers DictionaryLayer to fuse weights for inference
    device = cfg["device"]

    n_clips = 0
    sum_sq_res = 0.0
    sum_x = 0.0
    sum_sq_x = 0.0
    l0_total = 0.0
    feature_counts = torch.zeros(cfg["nb_concepts"], device=device)

    with torch.no_grad():
        for pixel_values in loader:
            pixel_values = pixel_values.to(device)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                videomae(pixel_values=pixel_values)

            activations = hook_storage["activations"].detach()

            for i in range(activations.shape[0]):
                tokens = activations[i].float()  # (1568, 768)
                _, codes, x_hat = sae(tokens)

                residual = tokens - x_hat
                sum_sq_res += residual.pow(2).sum().item()
                sum_x += tokens.sum().item()
                sum_sq_x += tokens.pow(2).sum().item()
                l0_total += (codes > 0).float().sum(-1).mean().item()
                feature_counts += (codes > 0).float().sum(0)
                n_clips += 1

    n_elements = n_clips * 1568 * cfg["input_dim"]
    mean_x = sum_x / n_elements
    var_x = sum_sq_x / n_elements - mean_x ** 2
    mse = sum_sq_res / n_elements
    r2 = 1.0 - mse / (var_x + 1e-8)
    l0 = l0_total / n_clips
    dead_features = int((feature_counts == 0).sum().item())

    sae.train()

    return {
        "val/r2": r2,
        "val/mse": mse,
        "val/l0": l0,
        "val/dead_features": dead_features,
        "_feature_counts": feature_counts.cpu(),  # for density histogram, stripped before wandb.log
    }


def main() -> None:
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
    wandb.init(project=CFG["wandb_project"], name=CFG["wandb_run"], config=CFG)

    print("Building data split...")
    train_paths, val_paths = build_split(CFG)

    print(f"Loading model: {CFG['model_id']}")
    videomae, processor, hook_storage = setup_videomae(CFG)

    train_loader, val_loader = build_loaders(train_paths, val_paths, processor, CFG)

    print("Setting up SAE...")
    sae, optimizer = setup_sae(CFG)
    print(f"  BatchTopKSAE: {CFG['input_dim']}d → {CFG['nb_concepts']} features, top_k={CFG['top_k']:,}")

    global_step = 0
    for epoch in range(CFG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CFG['epochs']}")

        avg_loss, global_step = train_epoch(
            sae, train_loader, videomae, hook_storage, optimizer, CFG, epoch, global_step
        )
        print(f"  Train loss: {avg_loss:.4f}")

        metrics = validate(sae, val_loader, videomae, hook_storage, CFG)
        feature_counts = metrics.pop("_feature_counts")

        wandb.log({**metrics, "epoch": epoch + 1}, step=global_step)
        print(
            f"  R²={metrics['val/r2']:.4f}  MSE={metrics['val/mse']:.6f}"
            f"  L0={metrics['val/l0']:.1f}  Dead={metrics['val/dead_features']}"
        )

        torch.save(sae.state_dict(), CFG["checkpoint"])
        print(f"  Saved: {CFG['checkpoint']}")

    # Feature density histogram at end of training
    density = (feature_counts / feature_counts.sum()).numpy()
    wandb.log({"feature_density": wandb.Histogram(density)}, step=global_step)

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
