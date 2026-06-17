"""
Profile activation statistics for a given backbone and layer.

Computes and saves the per-dimension mean of patch-token activations over N clips.
This dim_mean is subtracted from activations before SAE training.

Usage:
    uv run python notebooks/profile_activations.py
    MODEL_NAME=timesformer SAE_LAYER=7 uv run python notebooks/profile_activations.py
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import MODEL_REGISTRY, SSv2ClipDataset, load_metadata

CFG = {
    "model_name":      "videomae",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",       str(ROOT / "data" / "ssv2_val_set")),
    "layer":           7,
    "n_clips":         2000,
    "batch_size":      8,
    "num_workers":     2,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}

for _key, _env, _cast in [
    ("model_name", "MODEL_NAME", str),
    ("layer",      "SAE_LAYER",  int),
]:
    if os.environ.get(_env):
        CFG[_key] = _cast(os.environ[_env])

assert CFG["model_name"] in MODEL_REGISTRY, (
    f"Unknown model_name {CFG['model_name']!r}. Valid: {list(MODEL_REGISTRY)}"
)
_model_cfg = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"] = _model_cfg["num_frames"]

_abbrev = {"videomae": "vmae", "timesformer": "tf"}[CFG["model_name"]]
CFG["dim_mean_out"] = str(
    ROOT / "outputs" / "sae" / f"{_abbrev}_layer{CFG['layer']}_dim_mean.pt"
)


def main():
    device     = CFG["device"]
    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    cls_offset = model_cfg["cls_offset"]

    print(f"Device:    {device}")
    print(f"Model:     {CFG['model_name']} ({model_cfg['checkpoint']})")
    print(f"Layer:     {CFG['layer']}  cls_offset={cls_offset}")
    print(f"Output:    {CFG['dim_mean_out']}")

    print("Loading model...")
    processor = model_cfg["processor_class"].from_pretrained(model_cfg["checkpoint"])
    model     = model_cfg["model_class"].from_pretrained(model_cfg["checkpoint"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hook_storage = {}

    def _hook(module, input, output):
        raw = output[0] if isinstance(output, tuple) else output
        hook_storage["activations"] = raw[:, cls_offset:].detach()

    model_cfg["layer_getter"](model, CFG["layer"]).register_forward_hook(_hook)

    print("Loading clips...")
    _, clips, _ = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])
    paths = [video_dir / f"{c['id']}.webm" for c in clips]
    paths = [p for p in paths if p.exists()][: CFG["n_clips"]]
    print(f"  Using {len(paths)} clips")

    dataset = SSv2ClipDataset(paths, processor, CFG["num_frames"])
    loader  = DataLoader(
        dataset, batch_size=CFG["batch_size"], shuffle=False,
        num_workers=CFG["num_workers"],
    )

    all_acts = []
    print("Running forward passes...")
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                model(pixel_values=batch)
            all_acts.append(hook_storage["activations"].float().cpu())

    acts = torch.cat(all_acts, dim=0)   # (N, num_patch_tokens, hidden_dim)
    flat = acts.reshape(-1, acts.shape[-1])
    flat_all = acts.reshape(-1)

    n_tokens = acts.shape[0] * acts.shape[1]
    print(f"\n--- {CFG['model_name']} layer-{CFG['layer']} activation statistics ---")
    print(f"Shape: {tuple(acts.shape)}  ({n_tokens:,} patch tokens total)")
    print(f"Global scalar mean:  {flat_all.mean():.4f}")
    print(f"Global scalar std:   {flat_all.std():.4f}")
    print(f"Global min / max:    {flat_all.min():.3f} / {flat_all.max():.3f}")

    per_dim_mean = flat.mean(0)   # (hidden_dim,)
    per_dim_std  = flat.std(0)
    ratio = per_dim_mean.abs().mean() / per_dim_std.mean()
    print(f"\nPer-dim mean — min: {per_dim_mean.min():.4f},  max: {per_dim_mean.max():.4f},  "
          f"abs-mean: {per_dim_mean.abs().mean():.4f},  std: {per_dim_mean.std():.4f}")
    print(f"Per-dim std  — min: {per_dim_std.min():.4f},  max: {per_dim_std.max():.4f},  "
          f"mean: {per_dim_std.mean():.4f}")
    print(f"Ratio |mean| / std:  {ratio:.4f}  (>0.3 suggests mean-subtraction is worthwhile)")

    per_token_mean      = acts.mean(0)              # (num_patch_tokens, hidden_dim)
    token_mean_norm     = per_token_mean.norm(dim=-1)
    print(f"\nPer-token mean norm — min: {token_mean_norm.min():.3f}, "
          f"max: {token_mean_norm.max():.3f}, mean: {token_mean_norm.mean():.3f}")

    out_path = Path(CFG["dim_mean_out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(per_dim_mean.float(), out_path)
    print(f"\nSaved per-dim mean → {out_path}")


if __name__ == "__main__":
    main()
