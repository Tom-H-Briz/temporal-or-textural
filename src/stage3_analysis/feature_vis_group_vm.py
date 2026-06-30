"""
feature_vis_group_vm.py — VideoMAE spatiotemporal SAE feature activation maps.

Conditions: R (real), C1 (shuffled tubelet pairs), A (midpoint frame frozen).

Each image: 3 rows (clips) × 8 columns (one frame per tubelet).
VideoMAE: 16 frames in, 8 tubelet tokens × 14×14 patches = 1568 tokens.
Display: frames[::2] — first frame of each tubelet pair.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from sae import BatchTopKSAE
from ToT_utils import MODEL_REGISTRY, load_metadata, _strip_brackets

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------ß

CFG = {
    "model_flag":   "videomae",
    "class_id":     59,
    "features":     [3513, 3558, 5578, 1990],
    "n_clips":      3,
    "seed":         11,
    "layer":        7,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "video_dir":    Path("data/ssv2/20bn-something-something-v2"),
    "labels_path":  Path("data/ssv2/labels/labels.json"),
    "val_path":     Path("data/ssv2/labels/validation.json"),
    "num_frames":   16,     # frames fed to VideoMAE processor
    "num_tubelets": 8,      # temporal tokens (16 frames / tubelet_size 2) — display columns
    "n_spatial":    196,    # 14×14 spatial patches per tubelet
}

# ---------------------------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------------------------

def _resolve_cfg(cfg: dict) -> dict:
    sae_path = ROOT / "outputs" / "sae" / "sae_layer7_job64.pt"
    dim_mean = ROOT / "outputs" / "sae" / "layer7_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"VM SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": 64}

# ---------------------------------------------------------------------------
# CLIP LOADING
# ---------------------------------------------------------------------------

def load_clip_ids(val_path: Path, labels_path: Path, class_id: int) -> list[str]:
    label_map, clips, _ = load_metadata(str(labels_path), str(val_path))
    return [
        str(c["id"]) for c in clips
        if label_map.get(_strip_brackets(c["template"])) == class_id
    ]


def load_video_frames(clip_id: str, video_dir: Path, num_frames: int) -> list:
    """Load num_frames uniformly sampled numpy RGB arrays from clip."""
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


def c1_frames(frames: list, seed: int) -> list:
    """Shuffle consecutive tubelet pairs — preserves local pair structure."""
    pairs = [(frames[i], frames[i + 1]) for i in range(0, len(frames), 2)]
    order = np.random.default_rng(seed).permutation(len(pairs)).tolist()
    return [f for i in order for f in pairs[i]]


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

def extract_activations(frames: list, model, processor, sae,
                        dim_mean: torch.Tensor, cfg: dict, device: str) -> torch.Tensor:
    """Returns z: (num_patch_tokens, dict_size) — CLS excluded."""
    model_cfg  = MODEL_REGISTRY[cfg["model_flag"]]
    cls_offset = model_cfg["cls_offset"]

    pixel_values = processor(frames, return_tensors="pt")["pixel_values"].to(device)
    captured = {}

    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(hook_fn)
    with torch.no_grad():
        model(pixel_values=pixel_values)
    handle.remove()

    patch_hidden = captured["hidden"][0, cls_offset:, :]
    patch_hidden = patch_hidden - dim_mean
    _, z = sae.encode(patch_hidden.float())
    return z.detach().cpu()


def get_decoder_weights(sae) -> torch.Tensor:
    return sae.dictionary._weights.detach().cpu()

# ---------------------------------------------------------------------------
# SIGNED ACTIVATION MAP
# ---------------------------------------------------------------------------

def signed_activation_map(
    z: torch.Tensor,
    feature_idx: int,
    W_dec: torch.Tensor,
    num_tubelets: int,
    n_spatial: int,
    sign_override: float | None = None,
) -> np.ndarray:
    """Returns (num_tubelets, 14, 14) signed activations."""
    feat_acts = z[:, feature_idx]
    if sign_override is not None:
        dec_sign = float(np.sign(sign_override)) or 1.0
    else:
        dec_sign = float(torch.sign(W_dec[feature_idx].mean())) or 1.0
    signed    = feat_acts * dec_sign
    spatial_size = int(n_spatial ** 0.5)
    return signed.numpy().reshape(num_tubelets, spatial_size, spatial_size)

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def _overlay_frame(frame, act_patch, norm, cmap, alpha=0.45):
    H, W = frame.shape[:2]
    row_idx  = np.arange(H) * act_patch.shape[0] // H
    col_idx  = np.arange(W) * act_patch.shape[1] // W
    act_full = act_patch[np.ix_(row_idx, col_idx)]
    heat     = cmap(norm(act_full))[:, :, :3]
    blended  = frame / 255.0 * (1 - alpha) + heat * alpha
    return (blended * 255).clip(0, 255).astype(np.uint8)


def make_feature_image(clips_activations, clips_frames, clip_ids,
                       feature_idx, class_id, condition, output_path,
                       num_cols, vmax=None):
    n_clips = len(clips_activations)
    fig, axes = plt.subplots(n_clips, num_cols, figsize=(num_cols * 2, n_clips * 2.2))

    if vmax is None:
        all_vals = np.concatenate([a.flatten() for a in clips_activations])
        vmax = float(np.abs(all_vals).max())
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = cm.RdBu_r

    for row, (acts, frames, clip_id) in enumerate(zip(clips_activations, clips_frames, clip_ids)):
        for col in range(num_cols):
            ax = axes[row, col] if n_clips > 1 else axes[col]
            ax.imshow(_overlay_frame(frames[col], acts[col], norm, cmap))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"Clip {clip_id}", fontsize=7, rotation=0, labelpad=60, va="center")

    fig.suptitle(f"Feature {feature_idx} — Class {class_id} — {condition} [VM]", fontsize=10, y=1.01)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02, label="signed activation")
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
    out_dir = ROOT / "outputs/analysis/visualisations_vm" / f"class_{cfg['class_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, sae, dim_mean = load_model_and_sae(cfg, resolved, device)
    W_dec = get_decoder_weights(sae)

    import pandas as pd
    ranking_csv = ROOT / "outputs/analysis/per_class_feature_delta_vm_c1" / f"class_{cfg['class_id']}_feature_ranking.csv"
    if ranking_csv.exists():
        _df = pd.read_csv(ranking_csv, usecols=["feature_idx", "sign_R"])
        sign_dict = dict(zip(_df["feature_idx"], _df["sign_R"].astype(float)))
        print(f"  Loaded DFA signs for {len(sign_dict)} features from {ranking_csv.name}")
    else:
        sign_dict = {}
        print("  No ranking CSV found — using decoder-weight sign fallback")

    all_clip_ids = load_clip_ids(ROOT / cfg["val_path"], ROOT / cfg["labels_path"], cfg["class_id"])
    rng      = random.Random(cfg["seed"])
    clip_ids = rng.sample(all_clip_ids, cfg["n_clips"])
    print(f"Selected clips: {clip_ids}")

    for feat_idx in cfg["features"]:
        print(f"\nProcessing feature {feat_idx}...")

        r_activations, c1_activations, a_activations = [], [], []
        r_display, c1_display, a_display = [], [], []

        for clip_id in clip_ids:
            seed   = int(clip_id) % 2**32
            frames = load_video_frames(clip_id, ROOT / cfg["video_dir"], cfg["num_frames"])
            f_c1   = c1_frames(frames, seed)
            f_a    = midpoint_frames(frames)

            z_r  = extract_activations(frames, model, processor, sae, dim_mean, cfg, device)
            z_c1 = extract_activations(f_c1,  model, processor, sae, dim_mean, cfg, device)
            z_a  = extract_activations(f_a,   model, processor, sae, dim_mean, cfg, device)

            sign_ov = sign_dict.get(feat_idx)
            r_activations.append(signed_activation_map(z_r,  feat_idx, W_dec, cfg["num_tubelets"], cfg["n_spatial"], sign_ov))
            c1_activations.append(signed_activation_map(z_c1, feat_idx, W_dec, cfg["num_tubelets"], cfg["n_spatial"], sign_ov))
            a_activations.append(signed_activation_map(z_a,  feat_idx, W_dec, cfg["num_tubelets"], cfg["n_spatial"], sign_ov))

            # One display frame per tubelet — first frame of each pair
            r_display.append(frames[::2])
            c1_display.append(f_c1[::2])
            a_display.append(f_a[::2])

        r_vals      = np.concatenate([a.flatten() for a in r_activations])
        shared_vmax = float(np.abs(r_vals).max()) or 1.0

        for activations, display, cond in [
            (r_activations,  r_display,  "R"),
            (c1_activations, c1_display, "C1"),
            (a_activations,  a_display,  "A"),
        ]:
            make_feature_image(
                clips_activations=activations,
                clips_frames=display,
                clip_ids=clip_ids,
                feature_idx=feat_idx,
                class_id=cfg["class_id"],
                condition=cond,
                output_path=out_dir / f"feature_{feat_idx}_class{cfg['class_id']}_{cond}.png",
                num_cols=cfg["num_tubelets"],
                vmax=shared_vmax,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
