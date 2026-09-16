"""
Temporal positional-encoding differentiation across backbones — VV (ViViT), TF, VM.

Hypothesis being tested (design-chat ledger): position-locked features feed off the
temporal positional encoding. Self-attention is permutation-equivariant and CLS/
frame pooling is order-blind, so temporal order can ONLY enter a video transformer
through its positional embeddings — their cross-position differentiation is the
model's full order-carrying capacity. A backbone whose temporal slots are
near-identical (ViViT: cos ~ 1.0000, the paper's temporally-repeated init, never
differentiated in fine-tuning) has no such signal for position-locked features to
feed off, making it a weak control for that analysis.

Method: per checkpoint, extract the per-temporal-position embedding vectors
(VM/ViViT: joint position embedding, temporal-major token layout, mean over the
196 patches of each temporal slot; TF: dedicated time_embeddings, frame-indexed),
then measure pairwise cosine across positions.

Outputs (outputs/analysis/pos_enc/):
  pos_enc_temporal_summary.csv        one row per checkpoint: mean/min pairwise cos,
                                      differentiation (1 - mean cos), norms
  pos_enc_temporal_pairwise_cos.csv   long format: full cos(pos_i, pos_j) matrices —
                                      per-position structure, for the position-lock
                                      follow-up (which slots are near-duplicates)

Usage: uv run python notebooks/Pos_Enc_VV_TF_VM.py   (CPU-only, ~1 min, no clips)
"""

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, N_SPATIAL

CFG = {
    # every (backbone, dataset) checkpoint the project has — the table is complete,
    # not a cherry-picked selection
    "checkpoints": [
        ("vivit",      "kinetics400"),
        ("videomae",   "ssv2"),
        ("videomae",   "kinetics400"),
        ("timesformer", "ssv2"),
    ],
    "output_dir": ROOT / "outputs" / "analysis" / "pos_enc",
}

# embeddings submodule per backbone — script-local (the registry stops at the
# encoder-block getter; this diagnostic needs the embeddings, a different seam)
EMBEDDINGS_GETTER = {
    "videomae":    lambda model: model.videomae.embeddings,
    "vivit":       lambda model: model.vivit.embeddings,
    "timesformer": lambda model: model.timesformer.embeddings,
}


def temporal_position_vectors(model_name: str, model) -> torch.Tensor:
    """(n_positions, D): the temporal order-carrier vectors for one checkpoint."""
    emb = EMBEDDINGS_GETTER[model_name](model)
    if model_name == "timesformer":
        return emb.time_embeddings[0]  # (num_frames, D), frame-indexed directly
    cfg = MODEL_REGISTRY[model_name]
    pos = emb.position_embeddings[0][cfg["cls_offset"]:]  # CLS excluded via registry
    n_pos = pos.shape[0] // N_SPATIAL
    return pos.reshape(n_pos, N_SPATIAL, -1).mean(dim=1)  # temporal-major layout


def cosine_matrix(vecs: torch.Tensor) -> torch.Tensor:
    norm = F.normalize(vecs, dim=-1)
    return norm @ norm.T


def offdiag_stats(cos: torch.Tensor) -> dict:
    off = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    return {
        "n_positions":       cos.shape[0],
        "mean_pairwise_cos": round(off.mean().item(), 4),
        "min_pairwise_cos":  round(off.min().item(), 4),
        "differentiation":   round(1.0 - off.mean().item(), 4),
    }


def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, pair_rows = [], []
    for model_name, dataset_name in CFG["checkpoints"]:
        cfg = MODEL_REGISTRY[model_name]
        checkpoint = CHECKPOINT_REGISTRY[(model_name, dataset_name)]
        model = cfg["model_class"].from_pretrained(checkpoint)

        vecs = temporal_position_vectors(model_name, model)
        cos = cosine_matrix(vecs)
        source = "time_embeddings" if model_name == "timesformer" else "joint_pos_emb"
        summary_rows.append({
            "model": model_name, "dataset": dataset_name, "source": source,
            **offdiag_stats(cos),
            "mean_norm": round(vecs.norm(dim=-1).mean().item(), 4),
        })
        for i in range(cos.shape[0]):
            for j in range(cos.shape[0]):
                pair_rows.append({"model": model_name, "dataset": dataset_name,
                                  "pos_i": i, "pos_j": j, "cosine": round(cos[i, j].item(), 4)})
        if model_name == "timesformer":  # magnitude context: temporal vs spatial signal
            emb = EMBEDDINGS_GETTER[model_name](model)
            spatial = emb.position_embeddings[0, 1:].norm(dim=-1).mean().item()
            print(f"  TF context: ||time|| {vecs.norm(dim=-1).mean():.3f} vs "
                  f"||spatial|| {spatial:.3f} (ratio {vecs.norm(dim=-1).mean()/spatial:.3f})")


    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "pos_enc_temporal_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nTemporal positional-encoding differentiation (per checkpoint):")
    print(summary.to_string(index=False))
    print(f"\nSaved: {summary_path}")

    pair_path = out_dir / "pos_enc_temporal_pairwise_cos.csv"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False)
    print(f"Saved: {pair_path}  ({len(pair_rows)} rows — full cos(pos_i, pos_j) matrices)")


if __name__ == "__main__":
    main()
