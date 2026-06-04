"""
Ablation validation for DFAEngine — clip 217421 (Tearing something just a little bit).

For the top 50 features by DFA score: zero each feature across all tokens,
run a no_grad forward pass with the ablated reconstruction, record logit drop.
Scatter: DFA score vs actual logit drop. Print Pearson r.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

CLIP_ID         = "22357"
TARGET_CLASS    = "Tearing something just a little bit"
TOP_K           = 50

LABELS_PATH     = ROOT / "data/ssv2/labels/labels.json"
VALIDATION_PATH = ROOT / "data/ssv2/labels/validation.json"
VIDEO_DIR       = ROOT / "data/ssv2/20bn-something-something-v2"
SAE_PATH        = ROOT / "outputs/sae/sae_layer7_job128.pt"
DIM_MEAN_PATH   = ROOT / "outputs/sae/layer7_dim_mean.pt"
OUTPUT_DIR      = Path(__file__).parent / "output"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    label_map, _, _ = load_metadata(str(LABELS_PATH), str(VALIDATION_PATH))
    correct_class_idx = label_map[TARGET_CLASS]
    clip_path = VIDEO_DIR / f"{CLIP_ID}.webm"
    print(f"Clip: {clip_path}")
    print(f"Correct class index: {correct_class_idx}\n")

    with DFAEngine(
        model_flag="videomae",
        sae_path=SAE_PATH,
        dim_mean_path=DIM_MEAN_PATH,
        layer=7,
        device=DEVICE,
    ) as engine:
        # DFA forward+backward
        result = engine.run(clip_path, correct_class_idx)
        print(
            f"Original — correct={result.correct} "
            f"pred={result.predicted_class} "
            f"logit={result.correct_class_logit:.4f}"
        )

        # z without backward (shape: num_tokens x dict_size)
        z = engine.get_z(clip_path)
        print(f"z shape: {z.shape}  dtype: {z.dtype}\n")

        # Top-50 features by DFA score
        dfa_scores = result.per_feature_summary.cpu()
        top_indices = dfa_scores.argsort(descending=True)[:TOP_K].tolist()

        # Preprocess once — reused for every ablation forward pass
        cfg_num_frames = 16
        from ToT_utils import SSv2ClipDataset
        import av

        container = av.open(str(clip_path))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        n = len(frames)
        indices = torch.linspace(0, n - 1, cfg_num_frames).long().tolist()
        sampled = [frames[i] for i in indices]
        pixel_values = engine._processor(sampled, return_tensors="pt")["pixel_values"].to(DEVICE)

        # Ablation loop
        print(f"Running {TOP_K} ablation forward passes...")
        ablated_logits = []

        with torch.no_grad():
            for feat_idx in top_indices:
                z_abl = z.clone()
                z_abl[:, feat_idx] = 0.0

                engine._z_override = z_abl
                output = engine._model(pixel_values=pixel_values)
                engine._z_override = None

                logit = output.logits[0, correct_class_idx].item()
                ablated_logits.append(logit)

        # Results
        original_logit = result.correct_class_logit
        actual_drops = [original_logit - abl for abl in ablated_logits]
        dfa_scores_top = [dfa_scores[i].item() for i in top_indices]

        print(f"\n{'Feature':>8}  {'DFA score':>10}  {'Logit drop':>10}")
        print("-" * 34)
        for feat_idx, score, drop in zip(top_indices, dfa_scores_top, actual_drops):
            print(f"{feat_idx:>8}  {score:>10.4f}  {drop:>10.4f}")

        r, p = pearsonr(dfa_scores_top, actual_drops)
        print(f"\nPearson r = {r:.4f}  (p = {p:.4e})")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dfa_scores_top, actual_drops, alpha=0.7, edgecolors="none", s=40)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("DFA score (grad × activation, summed over tokens)")
    ax.set_ylabel("Actual logit drop (original − ablated)")
    ax.set_title(
        f"Ablation validation — clip {CLIP_ID}\n"
        f"Top {TOP_K} features by DFA score  |  Pearson r = {r:.3f}"
    )
    out_path = OUTPUT_DIR / f"ablation_validation_{CLIP_ID}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
