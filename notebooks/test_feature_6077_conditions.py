"""
Feature 6077 across perturbation conditions — clip 217421 (Tearing just a little bit).

Runs DFAEngine.run() for R/A/B/C inside a single context manager block.
Correct class index is always 150 — we're asking how much feature 6077 contributes
to the tearing classification regardless of what the model predicts.
"""

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from stage3_analysis.dfa_engine import DFAEngine

CLIP_ID             = "217421"
CORRECT_CLASS_IDX   = 150
TARGET_FEATURE      = 6077

VIDEO_DIR           = ROOT / "data/ssv2/20bn-something-something-v2"
PERTURB_META        = ROOT / "data/perturbation_metadata.parquet"
SAE_PATH            = ROOT / "outputs/sae/sae_layer7_job128.pt"
DIM_MEAN_PATH       = ROOT / "outputs/sae/layer7_dim_mean.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"Clip: {CLIP_ID}  |  correct class index: {CORRECT_CLASS_IDX}  |  target feature: {TARGET_FEATURE}\n")

    meta = pd.read_parquet(PERTURB_META)
    meta["clip_id"] = meta["clip_id"].astype(str)
    row = meta[meta["clip_id"] == CLIP_ID]
    if row.empty:
        raise RuntimeError(f"Clip {CLIP_ID} not found in {PERTURB_META}")
    row = row.iloc[0]

    conditions = [
        ("R", VIDEO_DIR / f"{CLIP_ID}.webm"),
        ("A", Path(row["path_A"])),
        ("B", Path(row["path_B"])),
        ("C", Path(row["path_C"])),
    ]

    with DFAEngine(
        model_flag="videomae",
        sae_path=SAE_PATH,
        dim_mean_path=DIM_MEAN_PATH,
        layer=7,
        device=DEVICE,
    ) as engine:
        for label, clip_path in conditions:
            result = engine.run(clip_path, CORRECT_CLASS_IDX)

            dfa_scores = result.per_feature_summary   # (dict_size,) float32
            feat_score = dfa_scores[TARGET_FEATURE].item()
            # rank: 1 = highest DFA score
            rank = int((dfa_scores > feat_score).sum().item()) + 1

            print(
                f"Condition {label} | "
                f"correct={str(result.correct):<5} | "
                f"logit={result.correct_class_logit:>6.2f} | "
                f"H_norm={result.entropy_normalised:.4f} | "
                f"feat_{TARGET_FEATURE} DFA={feat_score:.4f} | "
                f"rank={rank}"
            )


if __name__ == "__main__":
    main()
