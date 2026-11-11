"""
Which non-spike features resonate with VM's position spikes? Reads vm_qk_resonance.py's spike_pairs parquet.
Per head group, z is averaged over the group's heads; each feature keeps its best |z| over spikes (signed, with the spike's
tubelet), seek (feature = query, spike = key) and sought (spike = query, feature = key) separately.
Labels: "others" = same-tubelet near-misses kept out of the spike set; "pair" = VM pair-contrast feature (home class).
Usage: SAE_LAYER=5 uv run python src/stage3_analysis/vm_qk_partners.py
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "spikes": {5004: 0, 358: 1, 2093: 2, 3516: 3, 3938: 4, 917: 5, 449: 7, 5721: 6, 2179: 6, 935: 6, 5759: 6},
    "others": [1837, 3226, 223, 4349, 2383, 4763, 1550, 4410],
    "head_groups": {"same": [6, 9, 10, 11], "fwd": [1, 7], "back": [2]},   # fwd: query at t reads key at t+1
    "top_n": 20,
    "pairs_parquet": ROOT / f"outputs/analysis/vm_qk_resonance/spike_pairs_L{L}.parquet",
    "pair_csv": ROOT / f"outputs/analysis/videomae_pair_contrast_L{L}/features.csv",
    "out_dir": ROOT / "outputs/analysis/vm_qk_resonance",
}


def best_over_spikes(t, col, spikes):
    """Per feature: signed z with the largest |z| over spikes, and that spike as 't6 #5721'."""
    w = t.pivot_table(index="feature", columns="ramp", values=col)
    arg = w.abs().idxmax(axis=1)
    lab = [f"t{spikes[s]} #{s}" for s in arg]
    return pd.DataFrame({col: [w.at[f, s] for f, s in arg.items()], f"{col[:-2]}_spike": lab}, index=w.index)


def partners(cfg, tab, heads):
    """Non-spike features ranked by best |z| (seek or sought) for one head group."""
    t = tab[tab["head"].isin(heads) & tab.ramp.isin(cfg["spikes"]) & ~tab.feature.isin(cfg["spikes"])]
    t = t.groupby(["ramp", "feature"])[["seek_z", "sought_z"]].mean().reset_index()
    out = pd.concat([best_over_spikes(t, c, cfg["spikes"]) for c in ("seek_z", "sought_z")], axis=1)
    out["top"] = out[["seek_z", "sought_z"]].abs().max(axis=1)
    return out.sort_values("top", ascending=False)


def labels(cfg):
    """feature -> 'others' or 'pair:<home classes>' (pair-contrast features)."""
    p = pd.read_csv(cfg["pair_csv"]).groupby("latent").home.agg(lambda h: "pair:" + ",".join(map(str, sorted(set(h)))))
    lab = p.to_dict()
    lab.update({f: "others" for f in cfg["others"]})
    return lab


def main(cfg):
    tab, lab = pd.read_parquet(cfg["pairs_parquet"]), labels(cfg)
    pd.set_option("display.width", 200)
    for name, heads in cfg["head_groups"].items():
        p = partners(cfg, tab, heads).assign(label=lambda d: d.index.map(lambda f: lab.get(f, "-")))
        p.round(2).to_csv(cfg["out_dir"] / f"partners_{name}_L{L}.csv")
        print(f"\n=== {name} heads {heads}: top {cfg['top_n']} (99.9 pct top |z| = {p.top.quantile(.999):.1f})")
        print(p.head(cfg["top_n"]).round(1).to_string())


if __name__ == "__main__":
    main(CFG)
