"""
Decoder-weight cosine similarity for colliding position-locked feature pairs.

L5 position-locking shows multiple features landing on the same tubelet slot
(SSv2: 935/5721 both lock tubelet 6; K400: 3003/5377 at position 5, 3421/4534
at position 7). K400's collision pairs also favour largely-overlapping top-5
classes — reads as redundancy, not independent detectors. This tests that
directly: cosine similarity between each pair's SAE decoder weight vectors
(W_dec row = the direction that feature reconstructs in hidden space), plus
a random-pair baseline from the same dictionary so a given cosine value can
be read as unusual or not, not just eyeballed against +-1.

Outputs:
    outputs/analysis/position_lock/cosine_similarity_pairs.csv

Usage:
    uv run python src/stage3_analysis/cosine_similarity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import resolve_sae_checkpoint

CFG = {
    "layer": 5,
    "sae_k": 64,
    "n_random_pairs": 10_000,
    "seed": 0,
    "configs": {
        ("videomae", "ssv2"):        [(935, 5721)],
        ("videomae", "kinetics400"): [(3003, 5377), (3421, 4534)],
    },
}
OUT_PATH = ROOT / "outputs/analysis/position_lock/cosine_similarity_pairs.csv"


def load_decoder_weights(model_flag: str, dataset_name: str, layer: int, sae_k: int) -> torch.Tensor:
    """W_dec: (dict_size, hidden_dim) — no need to instantiate the SAE class or
    run a forward pass, the decoder direction is just the raw checkpoint weight."""
    resolved = resolve_sae_checkpoint(model_flag, layer, dataset_name=dataset_name, sae_k=sae_k)
    ckpt = torch.load(resolved["sae_path"], map_location="cpu", weights_only=True)
    state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
    return state_dict["dictionary._weights"].detach().float()


def cosine_sim(w: torch.Tensor, i: int, j: int) -> float:
    a, b = w[i], w[j]
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def random_pair_baseline(w: torch.Tensor, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Cosine similarity for n_pairs random (distinct) feature pairs from the
    same dictionary — the null distribution a tested pair's value should be
    read against, not raw +-1 intuition."""
    n_dict = w.shape[0]
    i = rng.integers(0, n_dict, n_pairs)
    j = rng.integers(0, n_dict, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    w_norm = w / w.norm(dim=1, keepdim=True)
    return (w_norm[i] * w_norm[j]).sum(dim=1).numpy()


def classify(cos: float) -> str:
    if cos >= 0.7:
        return "near-duplicate"
    if cos <= -0.7:
        return "near-antipodal (polar)"
    return "not simply related"


def main() -> None:
    rng = np.random.default_rng(CFG["seed"])
    rows = []
    for (model_flag, dataset_name), pairs in CFG["configs"].items():
        print(f"\n=== {model_flag} / {dataset_name} / L{CFG['layer']} ===")
        w = load_decoder_weights(model_flag, dataset_name, CFG["layer"], CFG["sae_k"])
        baseline = random_pair_baseline(w, CFG["n_random_pairs"], rng)
        print(f"  random-pair baseline (n={len(baseline)}): mean={baseline.mean():+.4f}  "
              f"std={baseline.std():.4f}  |cos| p99={np.percentile(np.abs(baseline), 99):.4f}")
        for i, j in pairs:
            cos = cosine_sim(w, i, j)
            z = (cos - baseline.mean()) / baseline.std()
            label = classify(cos)
            print(f"  ({i}, {j}): cosine={cos:+.4f}  z_vs_random={z:+.2f}  -> {label}")
            rows.append({"model_flag": model_flag, "dataset": dataset_name, "layer": CFG["layer"],
                        "feature_i": i, "feature_j": j, "cosine_similarity": cos,
                        "z_vs_random_baseline": z, "classification": label,
                        "baseline_mean": baseline.mean(), "baseline_std": baseline.std()})

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
