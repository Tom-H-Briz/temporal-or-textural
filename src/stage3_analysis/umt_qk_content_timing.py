"""
Content -> clock: QK resonance of the L5 pair-contrast content features with the 19 timing features
(both directions, z per head, as umt_qk_resonance.py). Each content feature's best |z| is ranked against
all 6,144 features. Temporal = auroc_Rev < 0.5 (separation swaps under reverse). Weights only.
Usage: uv run python src/stage3_analysis/umt_qk_content_timing.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "stage3_analysis"))
import umt_qk_resonance as qkr

CFG = dict(qkr.CFG, **{
    "pairs_csv": ROOT / f"outputs/analysis/umt_pair_contrast_L{qkr.L}/features.csv",
    "timing_csv": ROOT / f"outputs/analysis/umt_layer_positional/timing_features_L{qkr.L}.csv",
    "labels": ROOT / "data/ssv2/labels/labels.json",
    "out_dir": ROOT / "outputs/analysis/umt_qk_resonance",
})


def timing_z(cfg, timing):
    """seek / sought z, each (H, F, T): every feature vs every timing feature, per head."""
    q, k, _ = qkr.project(cfg, qkr.load_decoder(cfg), qkr.load_block(cfg))
    mu, sd = qkr.null_stats(cfg, q, k)
    t = torch.tensor(timing)
    seek = torch.einsum("fhd,thd->hft", q, k[t])                # feature = query, timing = key
    sought = torch.einsum("thd,fhd->hft", q[t], k)              # timing = query, feature = key
    z = lambda s: ((s - mu[:, None, None]) / sd[:, None, None]).numpy()
    return z(seek), z(sought)


def best(z):
    """Per feature: max |z| over heads x timing features, its signed value, head and timing index."""
    H, F, T = z.shape
    flat = np.abs(z).transpose(1, 0, 2).reshape(F, H * T)
    i = flat.argmax(1)
    return flat.max(1), z.transpose(1, 0, 2).reshape(F, H * T)[np.arange(F), i], i // T, i % T


def content_table(cfg, timing):
    """One row per content latent: its pair directions (home class names), min auroc_Rev, temporal flag."""
    import json
    names = {int(v): k[:28] for k, v in json.load(open(cfg["labels"])).items()}
    p = pd.read_csv(cfg["pairs_csv"])
    p = p[~p.latent.isin(timing)].assign(dir=lambda d: d.home.map(names))
    g = p.groupby("latent").agg(dirs=("dir", lambda s: " | ".join(sorted(set(s)))), rev=("auroc_Rev", "min"))
    return g.assign(temporal=g.rev < 0.5).reset_index().rename(columns={"latent": "feature"})


def main(cfg):
    tl = pd.read_csv(cfg["timing_csv"])
    timing = tl.feature.astype(int).tolist()
    ct = content_table(cfg, timing)
    rest = np.setdiff1d(np.arange(6144), timing)
    for name, z in zip(("seek", "sought"), timing_z(cfg, timing)):
        mx, val, h, ti = best(z)
        pct = (mx[rest][None, :] < mx[ct.feature.values][:, None]).mean(1) * 100
        ct[f"{name}_z"], ct[f"{name}_pct"] = val[ct.feature], pct
        ct[f"{name}_with"] = [f"#{timing[t]} h{hh}" for t, hh in zip(ti[ct.feature], h[ct.feature])]
    ct = ct.assign(top_pct=ct[["seek_pct", "sought_pct"]].max(1)).sort_values("top_pct", ascending=False)
    ct.round(2).to_csv(cfg["out_dir"] / f"content_vs_timing_L{cfg['sae_layer']}.csv", index=False)
    print(ct.drop(columns="top_pct").round(1).to_string(index=False))
    for col in ("seek_pct", "sought_pct"):
        print(f"\n{col}: median temporal {ct[ct.temporal][col].median():.0f} (n={ct.temporal.sum()}), "
              f"static {ct[~ct.temporal][col].median():.0f} (n={(~ct.temporal).sum()}), p = {perm_p(ct, col):.3f}")


def perm_p(ct, col, n=10_000, seed=0):
    """Two-sided permutation p for the temporal - static difference in median."""
    rng, x, lab = np.random.default_rng(seed), ct[col].values, ct.temporal.values
    d = lambda l: np.median(x[l]) - np.median(x[~l])
    obs = abs(d(lab))
    return (np.array([abs(d(rng.permutation(lab))) for _ in range(n)]) >= obs).mean()


if __name__ == "__main__":
    main(CFG)
