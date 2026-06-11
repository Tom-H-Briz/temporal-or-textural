"""
Manual feature ablation — zero a specific set of features simultaneously and
observe the logit change. Edit the config block and re-run.
"""

import sys
from pathlib import Path

import av
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ToT_utils import load_metadata
from stage3_analysis.dfa_engine import DFAEngine

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_ID  = "144528"
FEATURES = [3370,3130,2698,3196]          # all zeroed simultaneously; edit freely

CHIRAL_PATH     = ROOT / "outputs/analysis/chiral_extraction.parquet"
LABELS_PATH     = ROOT / "data/ssv2/labels/labels.json"
VALIDATION_PATH = ROOT / "data/ssv2/labels/validation.json"
VIDEO_DIR       = ROOT / "data/ssv2/20bn-something-something-v2"
SAE_PATH        = ROOT / "outputs/sae/sae_layer7_job128.pt"
DIM_MEAN_PATH   = ROOT / "outputs/sae/layer7_dim_mean.pt"
OUTPUT_DIR      = Path(__file__).parent / "output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────


def load_frames(clip_path: Path, num_frames: int = 16) -> list:
    container = av.open(str(clip_path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    indices = torch.linspace(0, len(frames) - 1, num_frames).long().tolist()
    return [frames[i] for i in indices]


WATCH_CLASSES = [93, 87]   # extra class indices to monitor alongside target


def forward(engine, pixel_values: torch.Tensor) -> torch.Tensor:
    output = engine._model(pixel_values=pixel_values)
    return output.logits[0].cpu()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    print(f"Clip:   {CLIP_ID}")
    print(f"Features to ablate: {FEATURES}\n")

    chiral = pd.read_parquet(CHIRAL_PATH)
    row = chiral.loc[chiral["clip_id"].astype(str) == CLIP_ID].iloc[0]
    target_class = row["class_name"]

    label_map, _, _ = load_metadata(str(LABELS_PATH), str(VALIDATION_PATH))
    class_idx = label_map[target_class]
    clip_path = VIDEO_DIR / f"{CLIP_ID}.webm"
    print(f"Class: {target_class}  (idx {class_idx})")

    watch = {idx: f"class {idx}" for idx in WATCH_CLASSES}
    watch[class_idx] = f"class {class_idx} (target)"

    with DFAEngine(
        model_flag="videomae",
        sae_path=SAE_PATH,
        dim_mean_path=DIM_MEAN_PATH,
        layer=7,
        device=DEVICE,
    ) as engine:
        z = engine.get_z(clip_path)
        print(f"z shape: {z.shape}  dtype: {z.dtype}")

        frames = load_frames(clip_path)
        pixel_values = engine._processor(frames, return_tensors="pt")["pixel_values"].to(DEVICE)

        with torch.no_grad():
            orig_logits = forward(engine, pixel_values)

            z_abl = z.clone()
            z_abl[:, FEATURES] = 0.0
            engine._z_override = z_abl
            abl_logits = forward(engine, pixel_values)
            engine._z_override = None

    print(f"\n{'Class':<28}  {'Original':>9}  {'Ablated':>9}  {'Drop':>9}")
    print("-" * 62)
    for idx, label in sorted(watch.items()):
        orig = orig_logits[idx].item()
        abl  = abl_logits[idx].item()
        print(f"{label:<28}  {orig:>9.4f}  {abl:>9.4f}  {orig - abl:>+9.4f}")

    print(f"\nFeatures zeroed: {FEATURES}")

    # Bar chart — one group per watched class
    indices = sorted(watch.keys())
    labels  = [watch[i] for i in indices]
    orig_vals = [orig_logits[i].item() for i in indices]
    abl_vals  = [abl_logits[i].item() for i in indices]

    x = range(len(indices))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar([i - width / 2 for i in x], orig_vals, width, label="original", color="steelblue")
    b2 = ax.bar([i + width / 2 for i in x], abl_vals,  width, label="ablated",  color="tomato")
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Logit")
    feature_str = str(FEATURES) if len(FEATURES) <= 5 else f"{len(FEATURES)} features"
    ax.set_title(f"Clip {CLIP_ID} — ablate {feature_str}")
    ax.legend()
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"feature_ablation_{CLIP_ID}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
