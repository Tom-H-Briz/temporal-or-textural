"""
feature_vis_vm.py — Single-feature, three-condition activation map (VideoMAE).

Rows: R (real) / C1 (shuffled pairs) / A (frozen midpoint)
Cols: 8 tubelets (one frame per tubelet)
Normalisation: global across all 3 conditions — intensity is directly comparable.

Set CLIP_ID to a specific clip, or leave None and set CLASS_ID for a random draw.
"""

import random
import sys
from pathlib import Path

import av
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from sae import BatchTopKSAE
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_metadata, _strip_brackets

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CFG = {
    "model_flag":   "videomae",
    "clip_id":      None,        # set to a specific clip ID string, or None to use class_id
    "class_id":     0,
    "feature_idx":  5384,
    "seed":         7,
    "layer":        7,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "video_dir":    Path("data/ssv2/20bn-something-something-v2"),
    "labels_path":  Path("data/ssv2/labels/labels.json"),
    "val_path":     Path("data/ssv2/labels/validation.json"),
    "num_frames":   16,
    "num_tubelets": 8,
    "n_spatial":    196,
    "sign_override": None,   # set to +1 or -1 to override decoder-weight sign (check signed_vec_R in parquet)
}

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

def _resolve_cfg(cfg: dict) -> dict:
    sae_path = ROOT / "outputs" / "sae" / "sae_layer7_job64.pt"
    dim_mean = ROOT / "outputs" / "sae" / "layer7_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"VM SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": 64}


def load_model_and_sae(cfg: dict, resolved: dict, device: str):
    model_cfg  = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], "ssv2")]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
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
# CLIP LOADING
# ---------------------------------------------------------------------------

def resolve_clip_id(cfg: dict) -> str:
    if cfg["clip_id"] is not None:
        return str(cfg["clip_id"])
    pq_ids = set(pd.read_parquet(
        ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1.parquet",
        columns=["clip_id"]
    )["clip_id"].tolist())
    label_map, clips, _ = load_metadata(
        str(ROOT / cfg["labels_path"]), str(ROOT / cfg["val_path"])
    )
    ids = [str(c["id"]) for c in clips
           if label_map.get(_strip_brackets(c["template"])) == cfg["class_id"]
           and str(c["id"]) in pq_ids]
    assert ids, f"No R-correct clips found for class {cfg['class_id']}"
    return random.Random(cfg["seed"]).choice(ids)


def load_frames(clip_id: str, cfg: dict) -> list:
    container = av.open(str(ROOT / cfg["video_dir"] / f"{clip_id}.webm"))
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n = len(all_frames)
    idx = np.linspace(0, n - 1, cfg["num_frames"], dtype=int)
    frames = [all_frames[i] for i in idx]
    while len(frames) < cfg["num_frames"]:
        frames.append(frames[-1])
    return frames[:cfg["num_frames"]]


def c1_frames(frames: list, seed: int) -> list:
    pairs = [(frames[i], frames[i + 1]) for i in range(0, len(frames), 2)]
    order = np.random.default_rng(seed).permutation(len(pairs)).tolist()
    return [f for i in order for f in pairs[i]]


def midpoint_frames(frames: list) -> list:
    return [frames[len(frames) // 2]] * len(frames)

# ---------------------------------------------------------------------------
# ACTIVATION EXTRACTION
# ---------------------------------------------------------------------------

def extract_z(frames: list, model, processor, sae, dim_mean, cfg: dict, device: str) -> torch.Tensor:
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

    patch_hidden = captured["hidden"][0, cls_offset:, :] - dim_mean
    _, z = sae.encode(patch_hidden.float())
    return z.detach().cpu()


def lookup_dfa_sign(clip_id: str, feat: int) -> float:
    src = ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1.parquet"
    row = pd.read_parquet(src, columns=["clip_id", "signed_vec_R"]).query("clip_id == @clip_id")
    if row.empty:
        raise KeyError(f"Clip {clip_id} not found in mass delta parquet — is it R-correct and in the SL subset?")
    val = float(np.asarray(row.iloc[0]["signed_vec_R"])[feat])
    sign = np.sign(val)
    print(f"  DFA sign for f{feat} clip {clip_id}: signed_vec_R={val:.4f}  sign={int(sign):+d}")
    return float(sign) or 1.0


def activation_map(z: torch.Tensor, feat: int, cfg: dict, clip_id: str) -> np.ndarray:
    override = cfg.get("sign_override")
    dec_sign = float(override) if override is not None else lookup_dfa_sign(clip_id, feat)
    signed   = z[:, feat] * dec_sign
    spatial  = int(cfg["n_spatial"] ** 0.5)
    return signed.numpy().reshape(cfg["num_tubelets"], spatial, spatial)

# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def _overlay(frame: np.ndarray, patch: np.ndarray, norm, cmap, alpha: float = 0.45) -> np.ndarray:
    H, W = frame.shape[:2]
    ri = np.arange(H) * patch.shape[0] // H
    ci = np.arange(W) * patch.shape[1] // W
    heat = cmap(norm(patch[np.ix_(ri, ci)]))[:, :, :3]
    return (frame / 255.0 * (1 - alpha) + heat * alpha).clip(0, 1)


COND_LABELS = {"clip": "Clip", "R": "Real", "C1": "Shuffle", "A": "Still"}


def make_figure(maps: dict, frames_by_cond: dict, cfg: dict, clip_id: str, norm, cmap) -> plt.Figure:
    conditions = ["R", "C1", "A"]
    n_t = cfg["num_tubelets"]
    fig, axes = plt.subplots(4, n_t, figsize=(n_t * 2, 9))
    fig.subplots_adjust(left=0.08, right=0.91, hspace=0.05, wspace=0.03)

    # Row 0 — raw clip, no overlay
    for col, frame in enumerate(frames_by_cond["R"][::2]):
        axes[0, col].imshow(frame)
        axes[0, col].axis("off")
    mid_y = axes[0, 0].get_position().y0 + axes[0, 0].get_position().height / 2
    fig.text(0.01, mid_y, COND_LABELS["clip"], fontsize=11, fontweight="bold",
             va="center", ha="left")

    # Rows 1–3 — conditions with activation overlay
    for row, cond in enumerate(conditions, start=1):
        act = maps[cond]
        display = frames_by_cond[cond][::2]
        for col in range(n_t):
            ax = axes[row, col]
            ax.imshow(_overlay(display[col], act[col], norm, cmap))
            ax.axis("off")
        mid_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        fig.text(0.01, mid_y, COND_LABELS[cond], fontsize=11, fontweight="bold",
                 va="center", ha="left")

    fig.suptitle(
        f"Feature {cfg['feature_idx']} — Class {cfg['class_id']} — Clip {clip_id} [VM]",
        fontsize=11,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    vmax = norm.vmax
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.015, pad=0.01)
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f"{-vmax:.1f}", "0", f"{vmax:.1f}"])
    cbar.set_label("signed activation", fontsize=9)
    return fig

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    resolved = _resolve_cfg(CFG)
    device   = CFG["device"]

    out_dir = ROOT / "outputs/analysis/vis_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, sae, dim_mean = load_model_and_sae(CFG, resolved, device)

    clip_id = resolve_clip_id(CFG)
    print(f"Clip: {clip_id}  Feature: {CFG['feature_idx']}")

    frames_R = load_frames(clip_id, CFG)
    frames_C1 = c1_frames(frames_R, seed=int(clip_id) % 2**32)
    frames_A  = midpoint_frames(frames_R)
    frames_by_cond = {"R": frames_R, "C1": frames_C1, "A": frames_A}

    maps = {}
    for cond, frames in frames_by_cond.items():
        z = extract_z(frames, model, processor, sae, dim_mean, CFG, device)
        maps[cond] = activation_map(z, CFG["feature_idx"], CFG, clip_id)

    # Scale anchored to R's 99th percentile — C1 and A render dimmer relative to R
    vmax = float(np.percentile(np.abs(maps["R"]), 99))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = cm.RdBu_r

    fig = make_figure(maps, frames_by_cond, CFG, clip_id, norm, cmap)
    out_path = out_dir / f"{clip_id}_f{CFG['feature_idx']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
