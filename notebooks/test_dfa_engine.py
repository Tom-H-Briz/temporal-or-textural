"""
Throwaway test for DFAEngine.

Loads 10 clips from "Tearing something just a little bit", runs engine.run()
on each, and prints per-clip diagnostics + post-run sanity checks.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

TARGET_CLASS = "Tearing something just a little bit"
N_CLIPS = 10

LABELS_PATH     = ROOT / "data/ssv2/labels/labels.json"
VALIDATION_PATH = ROOT / "data/ssv2/labels/validation.json"
VIDEO_DIR       = ROOT / "data/ssv2/20bn-something-something-v2"
SAE_PATH        = ROOT / "outputs/sae/sae_layer7_job128.pt"
DIM_MEAN_PATH   = ROOT / "outputs/sae/layer7_dim_mean.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print(f"Device: {DEVICE}")

    label_map, clips, _ = load_metadata(str(LABELS_PATH), str(VALIDATION_PATH))

    target_clips = [
        c for c in clips if _strip_brackets(c["template"]) == TARGET_CLASS
    ][:N_CLIPS]

    if not target_clips:
        raise RuntimeError(f"No clips found for class: {TARGET_CLASS!r}")

    print(f"Loaded {len(target_clips)} clips for '{TARGET_CLASS}'\n")

    correct_class_idx = label_map[TARGET_CLASS]
    print(f"Correct class index: {correct_class_idx}\n")
    print(f"{'='*72}")

    with DFAEngine(
        model_flag="videomae",
        sae_path=SAE_PATH,
        dim_mean_path=DIM_MEAN_PATH,
        layer=7,
        device=DEVICE,
    ) as engine:
        summaries = []

        for clip_info in target_clips:
            clip_id = str(clip_info["id"])
            clip_path = VIDEO_DIR / f"{clip_id}.webm"

            result = engine.run(clip_path, correct_class_idx)
            summaries.append(result.per_feature_summary)

            s = result.per_feature_summary
            nonzero_count = int((s > 0).sum().item())

            print(
                f"clip {clip_id:>9} | "
                f"correct={str(result.correct):<5} | "
                f"pred={result.predicted_class:>4} | "
                f"logit={result.correct_class_logit:>8.4f} | "
                f"summary min={s.min().item():.4f} "
                f"max={s.max().item():.4f} "
                f"mean={s.mean().item():.4f} "
                f"nonzero={nonzero_count}"
            )

        print(f"{'='*72}")
        print("\n--- Post-run sanity checks ---")

        # Gradient checks — must be inside the context manager while model/SAE still exist
        sae_has_grad = any(
            p.grad is not None for p in engine._sae.parameters()
        )
        model_has_grad = any(
            p.grad is not None for p in engine._model.parameters()
        )
        print(f"SAE weights have grad:     {sae_has_grad}  (expected False)")
        print(f"VideoMAE weights have grad: {model_has_grad}  (expected False)")

    print("\n--- Spot checks on per_feature_summary values ---")
    for i, s in enumerate(summaries):
        nan_count = int(s.isnan().sum().item())
        neg_count = int((s < 0).sum().item())
        clip_id = str(target_clips[i]["id"])
        print(
            f"clip {clip_id:>9} | "
            f"NaN={nan_count} (expected 0) | "
            f"negative={neg_count} (expected 0)"
        )


if __name__ == "__main__":
    main()
