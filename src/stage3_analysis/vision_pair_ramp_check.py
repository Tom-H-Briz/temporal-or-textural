"""Are a model's top-DFA up/down features positional ramps? (CPU, from the pair-contrast outputs)

For the top-N features by share of |contrast DFA| (umt_vision_pair_aggregate.py arrays), the
mean activation per temporal position (UMT/TF frame, VM tubelet), by condition x class:
  corr_<cond>_<cls>   Pearson r of the profile with position index (+1 = rising ramp)
  ramp_shuffle        min r over both classes in the shuffle condition (ramp tied to POSITION,
                      not content order, if high)
  gate                activation(favoured class) / activation(other class) in R, favoured =
                      the class with the higher mean activation (direction-gated if >> 1)
  rev_gate            same ratio under Rev with the class roles swapped (gate follows the
                      perceived direction if >> 1)
Usage: MODEL=umt|videomae|timesformer uv run python src/stage3_analysis/vision_pair_ramp_check.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from umt_vision_pair_aggregate import CFG as AGG

CFG = {**AGG, "top_n": 15}   # top_n after **AGG so it overrides the aggregator default


def position_profiles(feats, ids, cond, cfg):
    """(n_clips, n_feats, positions) summed activation per temporal position."""
    out = np.zeros((len(ids), len(feats), cfg["n_frames"]), np.float32)
    col = np.full(cfg["n_latents"], -1)
    col[list(feats)] = np.arange(len(feats))
    for i, cid in enumerate(ids):
        s = np.load(cfg["in_dir"] / "sparse" / cond / f"{cid}.npz")
        pos = s["pos"].astype(int) if "pos" in s else s["tok"].astype(int) // 196
        j = col[s["lat"].astype(int)]
        m = j >= 0
        np.add.at(out[i], (j[m], pos[m]), s["z"][m].astype(np.float32))
    return out


def ramp_r(profile):
    """Pearson r of a mean profile with position index (nan if flat)."""
    t = np.arange(len(profile))
    return float(np.corrcoef(t, profile)[0, 1]) if profile.std() > 0 else np.nan


def feature_rows(feats, prof, up, shuf):
    """One row per feature: ramp r per condition x class, shuffle survival and gating."""
    rows = []
    for j, f in enumerate(feats):
        m = {k: (prof[k][up, j].mean(0), prof[k][~up, j].mean(0)) for k in prof}
        r = {f"corr_{k}_{c}": ramp_r(m[k][i]) for k in prof for i, c in enumerate(("up", "down"))}
        hi = 0 if m["R"][0].mean() >= m["R"][1].mean() else 1        # favoured class in R
        gate = m["R"][hi].mean() / max(m["R"][1 - hi].mean(), 1e-6)
        rev_gate = m["Rev"][1 - hi].mean() / max(m["Rev"][hi].mean(), 1e-6)
        rows.append(dict(feature=int(f), favours=("up", "down")[hi], **r,
                         ramp_shuffle=min(r[f"corr_{shuf}_up"], r[f"corr_{shuf}_down"]),
                         gate=gate, rev_gate=rev_gate))
    return pd.DataFrame(rows)


def main():
    cfg = CFG
    A = np.load(cfg["in_dir"] / "clip_feature_arrays_R.npz")
    pc = pd.read_parquet(cfg["in_dir"] / "per_clip.parquet")
    up = (pc[pc.condition == "R"].set_index("clip_id").true_class.loc[A["clip_id"]] == cfg["pair"][0]).to_numpy()
    share = np.abs(A["dfa"]).mean(0) / np.abs(A["dfa"]).sum(1).mean()
    feats = np.argsort(-share)[: cfg["top_n"]]
    shuf = cfg["conditions"][1]
    prof = {k: position_profiles(feats, A["clip_id"], c, cfg) for k, c in (("R", "R"), (shuf, shuf), ("Rev", "Rev"))}
    df = feature_rows(feats, prof, up, shuf)
    ft = pd.read_csv(cfg["in_dir"] / "feature_table.csv").set_index("feature")
    df.insert(1, "dfa_share", share[feats])
    df.insert(2, "evidence", ft.evidence.loc[feats].to_numpy())
    df.insert(3, "auroc", ft.auroc.loc[feats].to_numpy())
    df.to_csv(cfg["in_dir"] / "ramp_check.csv", index=False)
    np.savez_compressed(cfg["in_dir"] / "ramp_check_profiles.npz", feats=feats, up=up, **prof)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
