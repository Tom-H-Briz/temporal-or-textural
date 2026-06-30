"""
visualise_features.py — Spatiotemporal SAE feature activation maps
Project: Temporal or Textural?
Date: 23 June 2026

For a given class and list of features, produces two images per feature:
  feature_{idx}_class{class_id}_R.png  — real video
  feature_{idx}_class{class_id}_C.png  — shuffled video

Each image: 3 rows (clips) × 8 columns (frames).
Activation = z[feature_idx] × sign(W_dec[feature_idx].mean())
Colourmap: RdBu diverging, centred at zero, consistent scale per image.
"""

import random
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CFG = {
    "model_flag":  "timesformer",
    "class_id":    164,
    "features":    [4688, 4473, 1990, 5578],
    "n_clips":     3,
    "seed":        11,
    "layer":       7,
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "video_dir":   Path("data/ssv2/20bn-something-something-v2"),
    "labels_path": Path("data/ssv2/labels/labels.json"),
    "val_path":    Path("data/ssv2/labels/validation.json"),
    "output_dir":  Path("outputs/analysis/visualisations") / "class_164",
    "num_frames":  8,      # TF input frames
    "cls_offset":  1,      # TF prepends CLS at position 0
    "n_spatial":   196,    # 14×14 spatial patches per frame
}

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from sae import BatchTopKSAE
from ToT_utils import MODEL_REGISTRY, load_metadata, _strip_brackets

# ---------------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------------

def _resolve_cfg(cfg: dict) -> dict:
    sae_dir = ROOT / "outputs" / "sae"
    layer   = cfg["layer"]
    matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job*_best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 TF checkpoint for layer {layer}, found: {matches}"
        )
    ckpt_path = matches[0]
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae_k     = ckpt.get("sae_k")
    if sae_k is None:
        raise ValueError(f"sae_k not in checkpoint {ckpt_path}")
    dim_mean = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(ckpt_path), "dim_mean_path": str(dim_mean), "sae_k": sae_k}

# ---------------------------------------------------------------------------
# CLIP LOADING
# ---------------------------------------------------------------------------

def load_clip_ids(val_path: Path, labels_path: Path, class_id: int) -> list[str]:
    """Return all validation clip IDs belonging to class_id."""
    label_map, clips, _ = load_metadata(str(labels_path), str(val_path))
    return [
        str(c["id"]) for c in clips
        if label_map.get(_strip_brackets(c["template"])) == class_id
    ]


def load_video_frames(clip_id: str, video_dir: Path, num_frames: int) -> list:
    """Load frames from webm clip. Returns list of numpy RGB arrays, uniformly sampled."""
    import av
    container = av.open(str(video_dir / f"{clip_id}.webm"))
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n = len(all_frames)
    indices = np.linspace(0, n - 1, num_frames, dtype=int)
    frames = [all_frames[i] for i in indices]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def shuffle_frames(frames: list, seed: int) -> list:
    """Shuffle frame list in-place copy with fixed seed."""
    rng = random.Random(seed)
    shuffled = frames.copy()
    rng.shuffle(shuffled)
    return shuffled


def midpoint_frames(frames: list) -> list:
    """All frames replaced with the midpoint frame."""
    return [frames[len(frames) // 2]] * len(frames)

# ---------------------------------------------------------------------------
# MODEL + SAE LOADING
# ---------------------------------------------------------------------------

def load_model_and_sae(cfg: dict, resolved: dict, device: str):
    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    processor = model_cfg["processor_class"].from_pretrained(model_cfg["checkpoint"])
    model     = model_cfg["model_class"].from_pretrained(model_cfg["checkpoint"])
    model.to(device).eval().requires_grad_(False)

    ckpt       = torch.load(resolved["sae_path"], weights_only=True, map_location=device)
    state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    top_k = resolved["sae_k"] * model_cfg["num_patch_tokens"]
    sae = BatchTopKSAE(input_shape=model_cfg["hidden_dim"],
                       nb_concepts=nb_concepts, top_k=top_k, device=device)
    sae.load_state_dict(state_dict)
    dim_mean = torch.load(resolved["dim_mean_path"], weights_only=True, map_location=device)
    sae.train()
    dummy = torch.zeros(model_cfg["num_patch_tokens"], model_cfg["hidden_dim"], device=device)
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval().requires_grad_(False)
    return model, processor, sae, dim_mean


# ---------------------------------------------------------------------------
# ACTIVATION EXTRACTION
# ---------------------------------------------------------------------------

def extract_activations(
    frames: list,
    model,
    processor,
    sae,
    dim_mean: torch.Tensor,
    cfg: dict,
    resolved: dict,
    device: str,
) -> torch.Tensor:
    """
    Run forward pass, extract SAE latent z for all patch tokens.
    Returns z: (num_patch_tokens, dict_size) — CLS excluded.
    """
    model_cfg  = MODEL_REGISTRY[cfg["model_flag"]]
    layer_idx  = cfg["layer"]
    cls_offset = model_cfg["cls_offset"]

    # Process frames — frames is already a list of numpy RGB arrays
    pixel_values = processor(frames, return_tensors="pt")["pixel_values"].to(device)

    # Hook to capture layer output
    captured = {}

    def hook_fn(module, input, output):
        # output may be tuple (hidden_state, ...) depending on TF version
        hidden = output[0] if isinstance(output, tuple) else output
        # shape: (batch, 1+n_patch, hidden) for TF; (batch, n_patch, hidden) for VideoMAE
        captured["hidden"] = hidden.detach()

    layer_module = model_cfg["layer_getter"](model, layer_idx)
    handle = layer_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(pixel_values=pixel_values)

    handle.remove()

    hidden = captured["hidden"]  # (1, 1569, 768) for TF
    # Exclude CLS
    patch_hidden = hidden[0, cls_offset:, :]  # (1568, 768)

    # Mean subtraction
    patch_hidden = patch_hidden - dim_mean  # broadcast

    # SAE encode — returns (pre_codes, z)
    _, z = sae.encode(patch_hidden.float())

    return z.detach().cpu()


def get_decoder_weights(sae) -> torch.Tensor:
    """Return W_dec: (dict_size, hidden_dim)."""
    return sae.dictionary._weights.detach().cpu()

# ---------------------------------------------------------------------------
# SIGNED ACTIVATION MAP
# ---------------------------------------------------------------------------

def signed_activation_map(
    z: torch.Tensor,       # (n_patch_tokens, dict_size)
    feature_idx: int,
    W_dec: torch.Tensor,   # (dict_size, hidden_dim)
    num_frames: int,
    n_spatial: int,
    sign_override: float | None = None,
) -> np.ndarray:
    """
    Returns array of shape (num_frames, 14, 14) with signed activations.
    Sign from sign_override when provided (e.g. sign(mean_s_R) from per-class CSV),
    otherwise falls back to sign(W_dec[feature_idx].mean()).
    """
    feat_acts = z[:, feature_idx]  # (n_patch_tokens,)
    if sign_override is not None:
        dec_sign = float(np.sign(sign_override)) or 1.0
    else:
        dec_sign = float(torch.sign(W_dec[feature_idx].mean())) or 1.0

    signed = feat_acts * dec_sign  # (n_patch_tokens,)

    # Reshape to (num_frames, 14, 14)
    spatial_size = int(n_spatial ** 0.5)  # 14
    signed_np = signed.numpy().reshape(num_frames, spatial_size, spatial_size)
    return signed_np

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def _overlay_frame(
    frame: np.ndarray,      # (H, W, 3) uint8
    act_patch: np.ndarray,  # (ph, pw) float, signed
    norm: mcolors.Normalize,
    cmap,
    alpha: float = 0.45,
) -> np.ndarray:
    """Nearest-neighbour block-fill upsample then pixel-level alpha blend."""
    H, W = frame.shape[:2]
    row_idx = np.arange(H) * act_patch.shape[0] // H
    col_idx = np.arange(W) * act_patch.shape[1] // W
    act_full = act_patch[np.ix_(row_idx, col_idx)]
    heat = cmap(norm(act_full))[:, :, :3]
    blended = frame / 255.0 * (1 - alpha) + heat * alpha
    return (blended * 255).clip(0, 255).astype(np.uint8)

def make_feature_image(
    clips_activations: list[np.ndarray],  # list of (num_frames, 14, 14), one per clip
    clips_frames: list[list],             # list of frame lists (RGB arrays), one per clip
    clip_ids: list[str],
    feature_idx: int,
    class_id: int,
    condition: str,
    output_path: Path,
    num_frames: int,
    vmax: float | None = None,
):
    n_clips = len(clips_activations)
    fig, axes = plt.subplots(
        n_clips, num_frames,
        figsize=(num_frames * 2, n_clips * 2.2),
    )

    if vmax is None:
        all_vals = np.concatenate([a.flatten() for a in clips_activations])
        vmax = float(np.abs(all_vals).max())
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.RdBu_r  # Red = positive (driver), Blue = negative (suppressor)

    for row, (acts, frames, clip_id) in enumerate(
        zip(clips_activations, clips_frames, clip_ids)
    ):
        for col in range(num_frames):
            ax = axes[row, col] if n_clips > 1 else axes[col]
            frame_rgb = frames[col]
            composite = _overlay_frame(frame_rgb, acts[col], norm, cmap)
            ax.imshow(composite)

            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    f"Clip {clip_id}",
                    fontsize=7,
                    rotation=0,
                    labelpad=60,
                    va="center",
                )

    fig.suptitle(
        f"Feature {feature_idx} — Class {class_id} — {condition}",
        fontsize=10,
        y=1.01,
    )

    # Colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02,
                 label="signed activation")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    cfg      = CFG
    resolved = _resolve_cfg(cfg)
    device   = cfg["device"]

    out_dir = ROOT / "outputs/analysis/visualisations" / f"class_{cfg['class_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + SAE
    model, processor, sae, dim_mean = load_model_and_sae(cfg, resolved, device)
    W_dec = get_decoder_weights(sae)

    # Load per-class DFA signs from ranking CSV if available
    import pandas as pd
    ranking_csv = ROOT / "outputs/analysis/per_class_feature_delta" / f"class_{cfg['class_id']}_feature_ranking.csv"
    if ranking_csv.exists():
        _df = pd.read_csv(ranking_csv, usecols=["feature_idx", "sign_R"])
        sign_dict = dict(zip(_df["feature_idx"], _df["sign_R"].astype(float)))
        print(f"  Loaded DFA signs for {len(sign_dict)} features from {ranking_csv.name}")
    else:
        sign_dict = {}
        print(f"  No ranking CSV found — using decoder-weight sign fallback")

    # Select clips
    all_clip_ids = load_clip_ids(
        ROOT / cfg["val_path"],
        ROOT / cfg["labels_path"],
        cfg["class_id"],
    )
    rng = random.Random(cfg["seed"])
    clip_ids = rng.sample(all_clip_ids, cfg["n_clips"])
    print(f"Selected clips: {clip_ids}")

    # Per feature
    for feat_idx in cfg["features"]:
        print(f"\nProcessing feature {feat_idx}...")

        r_activations = []
        c_activations = []
        a_activations = []
        r_frames_all  = []
        c_frames_all  = []
        a_frames_all  = []

        for clip_id in clip_ids:
            frames   = load_video_frames(clip_id, ROOT / cfg["video_dir"], cfg["num_frames"])
            shuffled = shuffle_frames(frames, seed=int(clip_id) % 2**32)
            frozen   = midpoint_frames(frames)

            z_r    = extract_activations(frames,   model, processor, sae, dim_mean, cfg, resolved, device)
            z_c    = extract_activations(shuffled, model, processor, sae, dim_mean, cfg, resolved, device)
            z_a    = extract_activations(frozen,   model, processor, sae, dim_mean, cfg, resolved, device)
            sign_ov = sign_dict.get(feat_idx)
            acts_r = signed_activation_map(z_r, feat_idx, W_dec, cfg["num_frames"], cfg["n_spatial"], sign_ov)
            acts_c = signed_activation_map(z_c, feat_idx, W_dec, cfg["num_frames"], cfg["n_spatial"], sign_ov)
            acts_a = signed_activation_map(z_a, feat_idx, W_dec, cfg["num_frames"], cfg["n_spatial"], sign_ov)

            r_activations.append(acts_r)
            c_activations.append(acts_c)
            a_activations.append(acts_a)
            r_frames_all.append(frames)
            c_frames_all.append(shuffled)
            a_frames_all.append(frozen)

        # Dynamic range fixed to R so C and A are directly comparable
        r_vals = np.concatenate([a.flatten() for a in r_activations])
        shared_vmax = float(np.abs(r_vals).max()) or 1.0

        # Plot R
        make_feature_image(
            clips_activations=r_activations,
            clips_frames=r_frames_all,
            clip_ids=clip_ids,
            feature_idx=feat_idx,
            class_id=cfg["class_id"],
            condition="R",
            output_path=out_dir / f"feature_{feat_idx}_class{cfg['class_id']}_R.png",
            num_frames=cfg["num_frames"],
            vmax=shared_vmax,
        )

        # Plot C
        make_feature_image(
            clips_activations=c_activations,
            clips_frames=c_frames_all,
            clip_ids=clip_ids,
            feature_idx=feat_idx,
            class_id=cfg["class_id"],
            condition="C",
            output_path=out_dir / f"feature_{feat_idx}_class{cfg['class_id']}_C.png",
            num_frames=cfg["num_frames"],
            vmax=shared_vmax,
        )

        # Plot A
        make_feature_image(
            clips_activations=a_activations,
            clips_frames=a_frames_all,
            clip_ids=clip_ids,
            feature_idx=feat_idx,
            class_id=cfg["class_id"],
            condition="A",
            output_path=out_dir / f"feature_{feat_idx}_class{cfg['class_id']}_A.png",
            num_frames=cfg["num_frames"],
            vmax=shared_vmax,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()