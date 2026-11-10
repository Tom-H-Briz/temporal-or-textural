"""
Feature-splitting check: decoder cosine of the QK clock candidates vs the selected L5 ramps,
plus each candidate's nearest dictionary neighbours. Usage: uv run python src/stage3_analysis/umt_qk_candidate_cosine.py
"""
import os
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "candidates": [int(f) for f in os.environ.get("FEATURES", "3982,3999,3718").split(",")],
    "n_neighbours": 8, "n_null": 200_000, "seed": 0,
    "sae_path": ROOT / f"outputs/sae/sae_umt_ssv2_k64_x8_l{L}_job7ep_best.pt",
    "sets_csv": ROOT / f"outputs/analysis/umt_layer_positional/sets_L{L}.csv",
}


def main(cfg):
    sd = torch.load(cfg["sae_path"], map_location="cpu", weights_only=True)["sae_state_dict"]
    D = torch.nn.functional.normalize(sd["dictionary._weights"].float(), dim=1)
    sets = pd.read_csv(cfg["sets_csv"]).set_index("feature")
    ramps = sets.index[sets.arm == "ramp"].tolist()
    g = torch.Generator().manual_seed(cfg["seed"])
    i, j = (torch.randint(len(D), (cfg["n_null"],), generator=g) for _ in range(2))
    null = (D[i] * D[j]).sum(1).abs()
    print(f"null |cos|: 99th pct {null.quantile(0.99):.3f}, 99.99th {null.quantile(0.9999):.3f}")
    C = D[cfg["candidates"]] @ D.T                                 # (n_cand, F)
    print("\ncosine, candidate x selected ramp\n"
          + pd.DataFrame(C[:, ramps].numpy(), index=cfg["candidates"], columns=ramps).round(2).to_string())
    for c, row in zip(cfg["candidates"], C):
        row[c] = 0
        top = row.abs().topk(cfg["n_neighbours"]).indices.tolist()
        t = pd.DataFrame({"feature": top, "cos": row[top].numpy().round(3)})
        print(f"\n#{c} nearest neighbours\n{t.join(sets[['arm', 'vtc_rank']], on='feature').to_string(index=False, na_rep='-')}")


if __name__ == "__main__":
    main(CFG)
