"""
Top features per ramp x head in both QK directions (from umt_qk_resonance.py), labelled with
the positional-selection arm / VTC rank where known.
Usage: SAE_LAYER=5 [HEADS=0,4,8] uv run python src/stage3_analysis/umt_qk_resonance_top.py
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
SETS = ROOT / f"outputs/analysis/umt_layer_positional/sets_L{L}.csv"
CFG = {
    "layer": L, "top_n": 10, "overlap_n": 20,
    "ramps": [3453, 5422, 5736] if L == 5 else pd.read_csv(SETS).query("arm == 'ramp'").feature.tolist(),
    "heads": [int(h) for h in os.environ.get("HEADS", "0,4,8").split(",")],
    "pairs": ROOT / f"outputs/analysis/umt_qk_resonance/ramp_pairs_L{L}.parquet",
    "sets_csv": SETS,
}


def top(sub, col, n):
    """n features with the largest |z| in col (sign kept)."""
    return sub.reindex(sub[col].abs().sort_values(ascending=False).index).head(n)


def side_by_side(sub, labels, n):
    """Rank-aligned seek / sought lists: feature, signed z, arm, VTC rank."""
    cols = []
    for col in ("seek_z", "sought_z"):
        t = top(sub, col, n)[["feature", col]].merge(labels, on="feature", how="left")
        cols.append(t.rename(columns=lambda c: f"{col[:-2]}_{c}" if c != col else col).reset_index(drop=True))
    return pd.concat(cols, axis=1)


def main(cfg):
    tab = pd.read_parquet(cfg["pairs"])
    tab = tab[tab.feature != tab.ramp]
    labels = pd.read_csv(cfg["sets_csv"])[["feature", "arm", "vtc_rank"]].fillna({"arm": "important"})
    hits = {"seek_z": [], "sought_z": []}
    for r in cfg["ramps"]:
        for h in cfg["heads"]:
            sub = tab[(tab.ramp == r) & (tab["head"] == h)]
            a, b = (set(top(sub, c, cfg["overlap_n"]).feature) for c in ("seek_z", "sought_z"))
            rho = sub[["seek_z", "sought_z"]].corr("spearman").iloc[0, 1]
            print(f"\n=== ramp #{r}  head {h}  | top-{cfg['overlap_n']} shared: {len(a & b)}"
                  f"  | spearman(seek, sought) = {rho:.2f}")
            print(side_by_side(sub, labels, cfg["top_n"]).round(1).to_string(na_rep="-"))
            for c in hits:
                hits[c] += top(sub, c, cfg["top_n"]).feature.tolist()
    for c, f in hits.items():
        n = pd.Series(f, name="feature").value_counts().rename("n_blocks").reset_index()
        nb = len(cfg["ramps"]) * len(cfg["heads"])
        n = n[n.n_blocks >= max(3, nb // 5)].merge(labels, on="feature", how="left")
        print(f"\n{c}: features in the top {cfg['top_n']} of >= {max(3, nb // 5)} of the {nb} blocks\n{n.to_string(na_rep='-')}")


if __name__ == "__main__":
    main(CFG)
