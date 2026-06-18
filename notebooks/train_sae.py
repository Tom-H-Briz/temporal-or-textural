"""
Train BatchTopK SAE on VideoMAE or TimeSformer layer residual stream.

SSv2 validation set, real clips only — 20k train / 4k val, random split.
Output: outputs/sae/sae_{model}_k{k}_x{exp}_l{layer}_job{label}.pt

REMEMBER!!!! set WANDB_API_KEY env var in job script before running!
"""

import os
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from sae import BatchTopKSAE
from sae.losses import top_k_auxiliary_loss, reanimation_regularizer
from ToT_utils import MODEL_REGISTRY, SSv2ClipDataset, load_metadata

CFG = {
    "model_name":      "videomae",   # overridable via MODEL_NAME env var
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",       str(ROOT / "data" / "ssv2_val_set")),
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "layer":           7,
    # Training
    "epochs":          5,
    "batch_size":      64,
    "lr":              1e-4,
    "train_clips":     20_000,
    "split_seed":      42,
    "num_workers":     4,
    # Output / tracking
    "job_label":       "A",
    "output_dir":      str(ROOT / "outputs" / "sae"),
    "loss_fn":         "aux",   # "aux" | "reanimation"
    "wandb_project":   "temporal-or-textural",
    "resume_from":     None,
}

SAE_CONFIG = {
    "k":         64,
    "expansion": 8,
    "alpha":     0.03,
}


# --- SLURM env var overrides ---
# Apply all overrides before touching the registry so model_name is final first.
for _key, _env, _cast in [
    ("model_name", "MODEL_NAME",    str),
    ("loss_fn",    "SAE_LOSS_FN",   str),
    ("job_label",  "SAE_JOB_LABEL", str),
    ("epochs",     "SAE_EPOCHS",    int),
    ("layer",      "SAE_LAYER",     int),
]:
    if os.environ.get(_env):
        CFG[_key] = _cast(os.environ[_env])

for _key, _env, _cast in [
    ("k",         "SAE_K",         int),
    ("alpha",     "SAE_ALPHA",     float),
    ("expansion", "SAE_EXPANSION", int),
]:
    if os.environ.get(_env):
        SAE_CONFIG[_key] = _cast(os.environ[_env])

# --- Registry lookup and derived values ---
# Strict sequence: override → validate → lookup → derive. No dict literal computes these.
assert CFG["model_name"] in MODEL_REGISTRY, (
    f"Unknown model_name {CFG['model_name']!r}. Valid: {list(MODEL_REGISTRY)}"
)
_model_cfg = MODEL_REGISTRY[CFG["model_name"]]
_abbrev                 = {"videomae": "vmae", "timesformer": "tf"}[CFG["model_name"]]
CFG["num_frames"]       = _model_cfg["num_frames"]
CFG["hidden_dim"]       = _model_cfg["hidden_dim"]
CFG["num_patch_tokens"] = _model_cfg["num_patch_tokens"]
CFG["nb_concepts"]      = SAE_CONFIG["expansion"] * _model_cfg["hidden_dim"]
CFG["top_k"]            = SAE_CONFIG["k"] * CFG["num_patch_tokens"]
CFG["aux_loss_coeff"]   = SAE_CONFIG["alpha"]
CFG["dim_mean_path"]    = os.environ.get("DIM_MEAN_PATH") or str(
    ROOT / "outputs" / "sae" / f"{_abbrev}_layer{CFG['layer']}_dim_mean.pt"
)


def build_loss_fn(cfg: dict):
    if cfg["loss_fn"] == "reanimation":
        penalty = cfg["aux_loss_coeff"]
        def _reanimation_loss(x, x_hat, pre_codes, codes, dictionary):
            mse = (x - x_hat).pow(2).mean()
            return mse + reanimation_regularizer(x, x_hat, pre_codes, codes, dictionary, penalty=penalty)
        return _reanimation_loss
    return partial(top_k_auxiliary_loss, penalty=cfg["aux_loss_coeff"])


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
    val_paths   = all_paths[cfg["train_clips"] : cfg["train_clips"] + 4_000]
    print(f"  Train: {len(train_paths):,}  Val: {len(val_paths):,}")
    return train_paths, val_paths


def build_loaders(
    train_paths: list[Path], val_paths: list[Path], processor, cfg: dict
) -> tuple[DataLoader, DataLoader]:
    train_ds = SSv2ClipDataset(train_paths, processor, cfg["num_frames"])
    val_ds   = SSv2ClipDataset(val_paths,   processor, cfg["num_frames"])

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


def setup_model(cfg: dict) -> tuple:
    model_cfg        = MODEL_REGISTRY[cfg["model_name"]]
    cls_offset       = model_cfg["cls_offset"]
    num_patch_tokens = cfg["num_patch_tokens"]
    hidden_dim       = cfg["hidden_dim"]

    processor = model_cfg["processor_class"].from_pretrained(model_cfg["checkpoint"])
    model     = model_cfg["model_class"].from_pretrained(model_cfg["checkpoint"])
    model.to(cfg["device"]).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_storage: dict = {}

    def _hook(module, input, output) -> None:
        raw = output[0] if isinstance(output, tuple) else output
        # Tier-1 asserts: validate registry num_patch_tokens against the actual model output.
        assert raw.shape[1] == num_patch_tokens + cls_offset and raw.shape[2] == hidden_dim, (
            f"Raw block output shape mismatch: "
            f"expected (B, {num_patch_tokens + cls_offset}, {hidden_dim}), got {tuple(raw.shape)}"
        )
        patched = raw[:, cls_offset:]
        assert patched.shape[1] == num_patch_tokens and patched.shape[2] == hidden_dim, (
            f"Post-slice shape mismatch: "
            f"expected (B, {num_patch_tokens}, {hidden_dim}), got {tuple(patched.shape)}"
        )
        hook_storage["activations"] = patched

    model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(_hook)
    return model, processor, hook_storage


def setup_sae(cfg: dict) -> tuple[BatchTopKSAE, Adam]:
    sae = BatchTopKSAE(
        input_shape=cfg["hidden_dim"],
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
    model,
    hook_storage: dict,
    optimizer: Adam,
    cfg: dict,
    epoch: int,
    global_step: int,
    dim_mean: torch.Tensor,
    loss_fn,
) -> tuple[float, int]:
    sae.train()
    device = cfg["device"]
    total_loss = 0.0

    for pixel_values in loader:
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                model(pixel_values=pixel_values)

        activations = hook_storage["activations"].detach() - dim_mean
        n_clips = activations.shape[0]

        optimizer.zero_grad()
        batch_loss = 0.0

        for i in range(n_clips):
            tokens = activations[i].float()
            pre_codes, codes, x_hat = sae(tokens)
            loss = loss_fn(tokens, x_hat, pre_codes, codes, sae.get_dictionary())
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
    model,
    hook_storage: dict,
    cfg: dict,
    dim_mean: torch.Tensor,
) -> dict:
    sae.eval()
    device           = cfg["device"]
    num_patch_tokens = cfg["num_patch_tokens"]
    hidden_dim       = cfg["hidden_dim"]

    n_clips      = 0
    sum_sq_res   = 0.0
    sum_x        = 0.0
    sum_sq_x     = 0.0
    l0_total     = 0.0
    feature_counts = torch.zeros(cfg["nb_concepts"], device=device)

    with torch.no_grad():
        for pixel_values in loader:
            pixel_values = pixel_values.to(device)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                model(pixel_values=pixel_values)

            activations = hook_storage["activations"].detach() - dim_mean

            for i in range(activations.shape[0]):
                tokens = activations[i].float()
                _, codes, x_hat = sae(tokens)

                residual = tokens - x_hat
                sum_sq_res += residual.pow(2).sum().item()
                sum_x      += tokens.sum().item()
                sum_sq_x   += tokens.pow(2).sum().item()
                l0_total   += (codes > 0).float().sum(-1).mean().item()
                feature_counts += (codes > 0).float().sum(0)
                n_clips += 1

    n_elements = n_clips * num_patch_tokens * hidden_dim
    mean_x = sum_x / n_elements
    var_x  = sum_sq_x / n_elements - mean_x ** 2
    mse    = sum_sq_res / n_elements
    r2     = 1.0 - mse / (var_x + 1e-8)
    l0     = l0_total / n_clips
    dead_features = int((feature_counts == 0).sum().item())

    sae.train()

    return {
        "val/r2":            r2,
        "val/mse":           mse,
        "val/l0":            l0,
        "val/dead_features": dead_features,
        "_feature_counts":   feature_counts.cpu(),
    }


def main() -> None:
    Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)

    _abbrev  = {"videomae": "vmae", "timesformer": "tf"}[CFG["model_name"]]
    run_name = (
        f"sae_{_abbrev}_k{SAE_CONFIG['k']}_x{SAE_CONFIG['expansion']}"
        f"_l{CFG['layer']}_job{CFG['job_label']}"
    )
    CFG["checkpoint"]      = str(Path(CFG["output_dir"]) / f"{run_name}.pt")
    CFG["best_checkpoint"] = str(Path(CFG["output_dir"]) / f"{run_name}_best.pt")

    _dim_mean_path = Path(CFG["dim_mean_path"])
    if not _dim_mean_path.exists():
        raise FileNotFoundError(
            f"dim_mean not found: {_dim_mean_path}\n"
            f"Run compute_dim_mean_sweep.sh for {CFG['model_name']} layer {CFG['layer']} first."
        )
    dim_mean = torch.load(_dim_mean_path, weights_only=True).to(CFG["device"])
    print(f"  Loaded dim_mean from {_dim_mean_path}  shape={tuple(dim_mean.shape)}")

    loss_fn = build_loss_fn(CFG)

    wandb.init(
        project=CFG["wandb_project"],
        name=run_name,
        config={**CFG, "sae": SAE_CONFIG},
        tags=[f"job_{CFG['job_label']}", CFG["model_name"]],
    )

    print("Building data split...")
    train_paths, val_paths = build_split(CFG)

    print(f"Loading model: {_model_cfg['checkpoint']}")
    model, processor, hook_storage = setup_model(CFG)

    train_loader, val_loader = build_loaders(train_paths, val_paths, processor, CFG)

    print("Setting up SAE...")
    sae, optimizer = setup_sae(CFG)
    print(
        f"  BatchTopKSAE: {CFG['hidden_dim']}d → {CFG['nb_concepts']} features, "
        f"k={SAE_CONFIG['k']} (top_k={CFG['top_k']:,})"
    )
    print(f"  Loss: {CFG['loss_fn']}  α={CFG['aux_loss_coeff']}  Job: {CFG['job_label']}")

    start_epoch = 0
    if CFG["resume_from"] is not None:
        ckpt = torch.load(CFG["resume_from"], map_location=CFG["device"])
        sae.load_state_dict(ckpt["sae_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        sae.running_threshold = ckpt.get("running_threshold")
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed from {CFG['resume_from']} — starting at epoch {start_epoch}")

    best_score  = float("-inf")
    global_step = 0
    for epoch in range(start_epoch, CFG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CFG['epochs']}")

        avg_loss, global_step = train_epoch(
            sae, train_loader, model, hook_storage, optimizer, CFG, epoch, global_step,
            dim_mean, loss_fn,
        )
        print(f"  Train loss: {avg_loss:.4f}")

        metrics = validate(sae, val_loader, model, hook_storage, CFG, dim_mean)
        feature_counts = metrics.pop("_feature_counts")

        wandb.log({**metrics, "epoch": epoch + 1}, step=global_step)
        print(
            f"  R²={metrics['val/r2']:.4f}  MSE={metrics['val/mse']:.6f}"
            f"  L0={metrics['val/l0']:.1f}  Dead={metrics['val/dead_features']}"
        )

        ckpt_payload = {
            "epoch":                epoch,
            "sae_state_dict":       sae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "running_threshold":    sae.running_threshold,
        }
        epoch_ckpt = str(Path(CFG["output_dir"]) / f"{run_name}_epoch{epoch + 1}.pt")
        torch.save(ckpt_payload, epoch_ckpt)
        torch.save(ckpt_payload, CFG["checkpoint"])  # rolling latest for resume

        score = metrics["val/r2"] - (metrics["val/dead_features"] / CFG["nb_concepts"])
        wandb.log({"val/score": score, "epoch": epoch + 1}, step=global_step)
        if score > best_score:
            best_score = score
            torch.save(ckpt_payload, CFG["best_checkpoint"])
            print(f"  Saved epoch {epoch + 1}  ★ new best (score={score:.4f})")
        else:
            print(f"  Saved epoch {epoch + 1}  (best score={best_score:.4f})")

    val_tokens   = len(val_paths) * CFG["num_patch_tokens"]
    firing_rate  = feature_counts.float() / val_tokens
    fig, ax = plt.subplots()
    ax.hist(firing_rate.numpy(), bins=50, log=True)
    ax.set_xlabel("Fraction of val tokens")
    ax.set_ylabel("Number of features (log scale)")
    wandb.log({"feature_firing_freq": wandb.Image(fig)}, step=global_step)
    plt.close(fig)

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
