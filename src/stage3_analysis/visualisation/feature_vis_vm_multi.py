"""
feature_vis_vm_multi.py — Multi-feature, two-condition scaffold comparison (VideoMAE).

Columns: one per L5 position-locked feature, each shown only at its own locked
tubelet (not all 8, since each feature already owns a distinct one).
Rows: R (real) / C1 (shuffled pairs).
Column titles carry feature id, locked tubelet, and per-clip AP (% of the
clip's total abs DFA mass in R contributed by that feature).
"""

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

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from sae import BatchTopKSAE
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, resolve_sae_checkpoint

CFG = {
    "model_flag":   "videomae",
    "clip_id":      "161120",
    "class_id":     84,
    "layer":        5,
    "sae_k":        64,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "video_dir":    Path("data/ssv2/20bn-something-something-v2"),
    "num_frames":   16,
    "num_tubelets": 8,
    "n_spatial":    196,
    # (feature_idx, tubelet_idx) — L5_x8k64_VM scaffold members (16/09 gate)
    "features": [
        (5004, 0), (358, 1), (2093, 2), (3516, 3),
        (3938, 4), (917, 5), (449, 7),
    ],
}

DFA_PARQUET = (ROOT / "outputs/analysis/dfa_mass_delta_vm_c1" /
              "dfa_mass_delta_vm_c1_l{layer}_job7ep_k64.parquet")


def load_model_and_sae(cfg: dict):
    model_cfg  = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], "ssv2")]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(cfg["device"]).eval().requires_grad_(False)
    resolved   = resolve_sae_checkpoint(cfg["model_flag"], cfg["layer"], dataset_name="ssv2", sae_k=cfg["sae_k"])
    ckpt       = torch.load(resolved["sae_path"], weights_only=True, map_location=cfg["device"])
    state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    top_k = resolved["sae_k"] * model_cfg["num_patch_tokens"]
    sae = BatchTopKSAE(input_shape=model_cfg["hidden_dim"], nb_concepts=nb_concepts,
                       top_k=top_k, device=cfg["device"])
    sae.load_state_dict(state_dict)
    dim_mean = torch.load(resolved["dim_mean_path"], weights_only=True, map_location=cfg["device"])
    sae.train()
    dummy = torch.zeros(model_cfg["num_patch_tokens"], model_cfg["hidden_dim"], device=cfg["device"])
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval().requires_grad_(False)
    return model, processor, sae, dim_mean


def load_frames(clip_id: str, cfg: dict) -> list:
    container = av.open(str(ROOT / cfg["video_dir"] / f"{clip_id}.webm"))
    all_frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    idx = np.linspace(0, len(all_frames) - 1, cfg["num_frames"], dtype=int)
    return [all_frames[i] for i in idx]


def c1_frames(frames: list, seed: int) -> list:
    pairs = [(frames[i], frames[i + 1]) for i in range(0, len(frames), 2)]
    order = np.random.default_rng(seed).permutation(len(pairs)).tolist()
    return [f for i in order for f in pairs[i]]


def extract_z(frames: list, model, processor, sae, dim_mean, cfg: dict) -> torch.Tensor:
    model_cfg  = MODEL_REGISTRY[cfg["model_flag"]]
    cls_offset = model_cfg["cls_offset"]
    pixel_values = processor(frames, return_tensors="pt")["pixel_values"].to(cfg["device"])
    captured = {}

    def hook_fn(module, input, output):
        captured["hidden"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(hook_fn)
    with torch.no_grad():
        model(pixel_values=pixel_values)
    handle.remove()

    patch_hidden = captured["hidden"][0, cls_offset:, :] - dim_mean
    _, z = sae.encode(patch_hidden.float())
    return z.detach().cpu()


def load_clip_dfa_row(clip_id: str, layer: int) -> pd.Series:
    path = Path(str(DFA_PARQUET).format(layer=layer))
    cols = ["clip_id", "total_abs_R", "signed_vec_R", "signed_vec_C1"]
    df = pd.read_parquet(path, columns=cols)
    return df[df.clip_id == clip_id].iloc[0]


def activation_patch(z: torch.Tensor, feat: int, tubelet: int, dec_sign: float, cfg: dict) -> np.ndarray:
    spatial = int(cfg["n_spatial"] ** 0.5)
    grid = z[:, feat].numpy().reshape(cfg["num_tubelets"], spatial, spatial)
    return grid[tubelet] * dec_sign


def _overlay(frame: np.ndarray, patch: np.ndarray, norm, cmap, alpha: float = 0.45) -> np.ndarray:
    H, W = frame.shape[:2]
    ri = np.arange(H) * patch.shape[0] // H
    ci = np.arange(W) * patch.shape[1] // W
    heat = cmap(norm(patch[np.ix_(ri, ci)]))[:, :, :3]
    return (frame / 255.0 * (1 - alpha) + heat * alpha).clip(0, 1)


def make_figure(patches: dict, disp: dict, ap_pct: dict, columns: list, cfg: dict, norm, cmap) -> plt.Figure:
    n_col = len(columns)
    fig, axes = plt.subplots(2, n_col, figsize=(n_col * 2, 5))
    fig.subplots_adjust(left=0.06, right=0.9, hspace=0.05, wspace=0.05, top=0.82)

    for row, cond in enumerate(["R", "C1"]):
        for col, (feat, tub) in enumerate(columns):
            ax = axes[row, col]
            patch = patches[cond][col]
            if patch is None:
                ax.imshow(disp[cond][col])
            else:
                ax.imshow(_overlay(disp[cond][col], patch, norm, cmap))
            ax.axis("off")
            if row == 0:
                title = f"f{feat} (t{tub})\nAP {ap_pct[feat]:.2f}%" if feat is not None else f"t{tub}\n(no lock)"
                ax.set_title(title, fontsize=9)
        mid_y = axes[row, 0].get_position().y0 + axes[row, 0].get_position().height / 2
        label = {"R": "Real", "C1": "Shuffle"}[cond]
        fig.text(0.01, mid_y, label, fontsize=11, fontweight="bold", va="center", ha="left")

    fig.suptitle(f"L5 position-locked scaffold — Clip {cfg['clip_id']} (class {cfg['class_id']}) [VM]", fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    vmax = norm.vmax
    cax = fig.add_axes((0.92, 0.15, 0.015, 0.6))
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([-vmax, 0, vmax])
    cbar.set_ticklabels([f"{-vmax:.1f}", "0", f"{vmax:.1f}"])
    cbar.set_label("signed activation", fontsize=9)
    return fig


def main() -> None:
    out_dir = ROOT / "outputs/analysis/vis_single"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor, sae, dim_mean = load_model_and_sae(CFG)
    clip_id = CFG["clip_id"]
    dfa_row = load_clip_dfa_row(clip_id, CFG["layer"])

    frames_R = load_frames(clip_id, CFG)
    frames_C1 = c1_frames(frames_R, seed=int(clip_id) % 2**32)
    z_R = extract_z(frames_R, model, processor, sae, dim_mean, CFG)
    z_C1 = extract_z(frames_C1, model, processor, sae, dim_mean, CFG)

    tub_to_feat = {tub: feat for feat, tub in CFG["features"]}
    columns = [(tub_to_feat.get(tub), tub) for tub in range(CFG["num_tubelets"])]

    patches = {"R": [], "C1": []}
    disp = {"R": [], "C1": []}
    ap_pct = {}
    tot_abs_R = float(dfa_row["total_abs_R"])
    for feat, tub in columns:
        disp["R"].append(frames_R[tub * 2])
        disp["C1"].append(frames_C1[tub * 2])
        if feat is None:
            patches["R"].append(None)
            patches["C1"].append(None)
            continue
        sign_R  = float(np.sign(dfa_row["signed_vec_R"][feat])) or 1.0
        sign_C1 = float(np.sign(dfa_row["signed_vec_C1"][feat])) or 1.0
        patches["R"].append(activation_patch(z_R, feat, tub, sign_R, CFG))
        patches["C1"].append(activation_patch(z_C1, feat, tub, sign_C1, CFG))
        ap_pct[feat] = abs(float(dfa_row["signed_vec_R"][feat])) / tot_abs_R * 100

    all_patches = [p for p in patches["R"] + patches["C1"] if p is not None]
    vmax = float(max(np.abs(p).max() for p in all_patches))
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = cm.RdBu_r

    fig = make_figure(patches, disp, ap_pct, columns, CFG, norm, cmap)
    out_path = out_dir / f"{clip_id}_l5_scaffold_R_vs_shuffle.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
