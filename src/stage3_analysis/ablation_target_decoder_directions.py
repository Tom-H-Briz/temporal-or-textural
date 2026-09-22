"""
SAE-latent -> residual-stream direction for the L5 SSv2 ablation targets.

Each ablated latent's decode contribution is z[i] * dictionary[i, :] (see
sae/dictionary.py DictionaryLayer.forward: x_hat = z @ dictionary, rows are
per-concept directions, L2-normalised during training). This resolves what
residual-space direction run_ablation.py's per-feature ablation deltas
(ablation_summary_l5_job7ep_k64.csv, single_* targets) actually correspond
to, and whether the individually-ablated features share a direction or are
independent -- reusing load_decoder_weights/random_pair_baseline from
cosine_similarity.py rather than reloading the checkpoint a second way.

Outputs:
    outputs/analysis/scaffold_ablation/ablation_target_decoder_pairs_l5_job7ep_k64.csv

Usage:
    uv run python src/stage3_analysis/ablation_target_decoder_directions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_targets import get_targets, singleton_targets
from stage3_analysis.cosine_similarity import load_decoder_weights, random_pair_baseline

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "layer": 5,
    "sae_k": 64,
    "n_random_pairs": 10_000,
    "seed": 0,
    "summary_csv": ROOT / "outputs/analysis/scaffold_ablation/ablation_summary_l5_job7ep_k64.csv",
    "out_path": ROOT / "outputs/analysis/scaffold_ablation/ablation_target_decoder_pairs_l5_job7ep_k64.csv",
}


def load_single_deltas(summary_csv: Path) -> dict[int, dict[str, float]]:
    """feature_idx -> {mean_delta_R, mean_delta_C1} from run_ablation.py's
    overall (sl_label='overall') summary rows -- the per-feature ablation
    signal already computed, joined here to the direction each feature
    decodes to rather than recomputed."""
    df = pd.read_csv(summary_csv)
    overall = df[(df["sl_label"] == "overall") & df["ablation_target"].str.startswith("single_")]
    out: dict[int, dict[str, float]] = {}
    for _, row in overall.iterrows():
        idx = int(row["ablation_target"].removeprefix("single_"))
        out.setdefault(idx, {})[f"mean_delta_{row['perturbation_condition']}"] = row["mean_delta"]
    return out


def build_pair_rows(w: torch.Tensor, indices: list[int], deltas: dict[int, dict[str, float]],
                    baseline: np.ndarray) -> list[dict]:
    """One row per (feature_i, feature_j) among the ablation targets: decoder
    cosine similarity (shared residual direction, or not) plus each feature's
    own ablation delta, so a high cosine can be read against whether both
    features also damage the logit in the same direction."""
    w_norm = w[indices] / w[indices].norm(dim=1, keepdim=True)
    rows = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            cos = float((w_norm[a] * w_norm[b]).sum())
            rows.append({
                "feature_i": i, "feature_j": j, "cosine_similarity": cos,
                "z_vs_random_baseline": (cos - baseline.mean()) / baseline.std(),
                "delta_R_i": deltas.get(i, {}).get("mean_delta_R"),
                "delta_R_j": deltas.get(j, {}).get("mean_delta_R"),
                "delta_C1_i": deltas.get(i, {}).get("mean_delta_C1"),
                "delta_C1_j": deltas.get(j, {}).get("mean_delta_C1"),
            })
    return rows


def main() -> None:
    rng = np.random.default_rng(CFG["seed"])
    indices = [i for t in singleton_targets(get_targets(CFG["dataset"], CFG["layer"]))
              for i in get_targets(CFG["dataset"], CFG["layer"])[t]]

    w = load_decoder_weights(CFG["model_flag"], CFG["dataset"], CFG["layer"], CFG["sae_k"])
    baseline = random_pair_baseline(w, CFG["n_random_pairs"], rng)
    deltas = load_single_deltas(CFG["summary_csv"])

    rows = build_pair_rows(w, indices, deltas, baseline)
    print(f"L{CFG['layer']} {CFG['dataset']} — {len(indices)} ablation-target features, "
          f"{len(rows)} pairs  |  random-pair baseline mean={baseline.mean():+.4f} std={baseline.std():.4f}")
    for r in sorted(rows, key=lambda r: -abs(r["cosine_similarity"]))[:5]:
        print(f"  ({r['feature_i']}, {r['feature_j']}): cos={r['cosine_similarity']:+.4f}  "
              f"z={r['z_vs_random_baseline']:+.2f}")

    CFG["out_path"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CFG["out_path"], index=False)
    print(f"-> {CFG['out_path']}")


if __name__ == "__main__":
    main()
