"""
Do VM position-spike near-misses resonate with the spikes at their own tubelet? Reads vm_qk_resonance.py's outputs.
Pair score = mean over the same-tubelet heads of (seek_z + sought_z) / 2, so both QK directions count equally.
Per spike: score with the gate member at its tubelet, mean with the other same-tubelet spikes, mean with other tubelets.
Usage: SAE_LAYER=5 uv run python src/stage3_analysis/vm_qk_near_miss.py
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "heads": [6, 9, 10, 11],                                    # L5 same-tubelet heads (members: z 74-93)
    "res_dir": ROOT / "outputs/analysis/vm_qk_resonance",
    "sel_csv": ROOT / f"outputs/analysis/scaffold_selection/L{L}_x8k64_VM.csv",
}


def pair_matrix(cfg, sp):
    """Spike x spike symmetric score (heads mean of seek / sought averaged), self-pairs dropped."""
    t = pd.read_parquet(cfg["res_dir"] / f"spike_pairs_L{L}.parquet")
    t = t[t["head"].isin(cfg["heads"]) & t.feature.isin(sp.feature) & (t.feature != t.ramp)]
    t = t.assign(s=(t.seek_z + t.sought_z) / 2).groupby(["ramp", "feature"]).s.mean().unstack()
    return t.loc[sp.feature, sp.feature]


def spike_row(f, tub, M, sp):
    """One spike's score with its tubelet's member, its same-tubelet peers and the other tubelets."""
    same = sp[(sp.tub == tub) & (sp.feature != f)]
    mem = same[same.status == "member"].feature
    return {"to_member": M.loc[f, mem].mean() if len(mem) else float("nan"),
            "same_tub": M.loc[f, same.feature].mean() if len(same) else float("nan"),
            "other_tub": M.loc[f, sp[sp.tub != tub].feature].mean()}


def main(cfg):
    sp = pd.read_parquet(cfg["res_dir"] / f"spikes_L{L}.parquet")
    M = pair_matrix(cfg, sp)
    sel = pd.read_csv(cfg["sel_csv"]).rename(columns={"feature_idx": "feature"})
    sel = sel[["feature", "dfa_share_R", "dfa_share_C1", "z_consistency_R"]]
    rows = pd.DataFrame([spike_row(f, t, M, sp) for f, t in zip(sp.feature, sp.tub)])
    out = pd.concat([sp, rows], axis=1).merge(sel, on="feature", how="left")
    out.round(2).to_csv(cfg["res_dir"] / f"near_miss_L{L}.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.drop(columns="label").round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
