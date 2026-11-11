"""
VideoMAE QK resonance: the umt_qk_resonance.py method (He et al. 2024 Eq 8, weights only, z per head vs random
pairs) on VM's position spikes. The SAE hooks encoder.layer[L]'s output -> read by block L+1. VM has q_bias, no k bias.
Spikes = scaffold-selection gate members + near-misses, tagged with their tubelet (0-7). The parquet keeps
umt's "ramp" column name so the umt helpers work unchanged; here it holds the spike.
Usage: SAE_LAYER=5 uv run python src/stage3_analysis/vm_qk_resonance.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "stage3_analysis"))
import umt_qk_resonance as qkr                                  # project / null_stats / ramp_table / pull_table reused

L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "sae_layer": L, "attn_block": L + 1, "n_heads": 12, "n_null": 200_000, "seed": 0,
    "sae_path": ROOT / f"outputs/sae/sae_vmae_ssv2_k64_x8_l{L}_job7ep_best.pt",
    "spikes_csv": ROOT / f"outputs/analysis/scaffold_selection/L{L}_x8k64_VM.csv",
    "hf_checkpoint": "MCG-NJU/videomae-base-finetuned-ssv2",
    "out_dir": ROOT / "outputs/analysis/vm_qk_resonance",
}


def load_block(cfg):
    """Block attn_block's LayerNorm (γ, β), W_Q, W_K (out, in) and q_bias, same dict keys as qkr.load_block."""
    from transformers import VideoMAEForVideoClassification
    b = VideoMAEForVideoClassification.from_pretrained(cfg["hf_checkpoint"]).videomae.encoder.layer[cfg["attn_block"]]
    a = b.attention.attention
    assert a.q_bias is not None                                 # HF VideoMAE: k bias is fixed zeros
    f = lambda t: t.detach().float()
    return {"gamma": f(b.layernorm_before.weight), "beta": f(b.layernorm_before.bias),
            "Wq": f(a.query.weight), "Wk": f(a.key.weight), "bq": f(a.q_bias)}


def load_spikes(cfg):
    """Gate members + near-misses sorted by tubelet, labelled 't3 #3516' (* = near-miss)."""
    s = pd.read_csv(cfg["spikes_csv"]).query("status in ('member', 'near_miss')")
    s = s.assign(tub=s.z_position.astype(int), feature=s.feature_idx).sort_values(["tub", "status"])
    s["label"] = [f"t{t} #{f}{'*' * (st == 'near_miss')}" for t, f, st in zip(s.tub, s.feature, s.status)]
    return s[["feature", "tub", "status", "label"]].reset_index(drop=True)


def spike_grid(q, k, mu, sd, sp, heads):
    """Spike x spike z (row = query spike, column = key spike), mean over the given heads."""
    f = torch.tensor(sp.feature.values)
    z = (torch.einsum("ahd,bhd->hab", q[f], k[f]) - mu[:, None, None]) / sd[:, None, None]
    return pd.DataFrame(z[heads].mean(0).numpy(), index=sp.label, columns=sp.label).round(1)


def main(cfg):
    sp = load_spikes(cfg)
    q, k, pull = qkr.project(cfg, qkr.load_decoder(cfg), load_block(cfg))
    mu, sd = qkr.null_stats(cfg, q, k)
    tab = qkr.ramp_table(q, k, sp.feature.tolist(), mu, sd)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    for name, df in (("spike_pairs", tab), ("pull", qkr.pull_table(pull)), ("spikes", sp)):
        df.to_parquet(cfg["out_dir"] / f"{name}_L{cfg['sae_layer']}.parquet")
    print(f"spike x spike z, mean over heads (row = query, col = key)\n{spike_grid(q, k, mu, sd, sp, list(range(12))).to_string()}")
    for h in range(cfg["n_heads"]):
        print(f"\nhead {h}\n{spike_grid(q, k, mu, sd, sp, [h]).to_string()}")
    for col in ("seek_z", "sought_z"):
        print(f"\n{col}: 99th pct |z| over features, head x spike\n{qkr.head_spread(tab, col)}")


if __name__ == "__main__":
    main(CFG)
