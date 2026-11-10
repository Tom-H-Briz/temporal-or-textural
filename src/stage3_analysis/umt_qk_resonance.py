"""
QK resonance (He et al. 2024, Eq 8) between SAE features and the timing ramps, weights only.
S^h[f, g] = ď_f W_Q^h (ď_g W_K^h)^T, ď = LayerNorm-folded decoder direction (centred, γ-scaled).
  seek  = S[f, ramp]: feature f (query) looks for the ramp (key)
  sought = S[ramp, g]: the ramp (query) looks for feature g (key)
  pull  = (β W_Q + b_Q) W_K^T ď_g: every query's default pull towards g (UMT has no key bias)
Z-scored per head against random feature pairs. Ramps keep their own sign.

Usage: SAE_LAYER=5 uv run python src/stage3_analysis/umt_qk_resonance.py
"""
import os
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
L = int(os.environ.get("SAE_LAYER", 5))

CFG = {
    "sae_layer": L, "attn_block": L + 1,        # SAE hooks block L's output -> read by block L+1
    "n_heads": 12, "n_null": 200_000, "seed": 0,
    "sae_path": ROOT / f"outputs/sae/sae_umt_ssv2_k64_x8_l{L}_job7ep_best.pt",
    "sets_csv": ROOT / f"outputs/analysis/umt_layer_positional/sets_L{L}.csv",
    "hf_repo": "OpenGVLab/UMT", "hf_file": "multi_modality/ret_ssv2_tpl_b16_25m.pth",
    "out_dir": ROOT / "outputs/analysis/umt_qk_resonance",
}


def load_block(cfg):
    """Block attn_block's LayerNorm (γ, β), W_Q, W_K (out, in) and q_bias from the raw checkpoint."""
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download(cfg["hf_repo"], cfg["hf_file"]), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    p = f"vision_encoder.encoder.blocks.{cfg['attn_block']}."
    W = sd[p + "attn.qkv.weight"].float()                       # (3*768, 768): Q, K, V rows
    D = W.shape[1]
    return {"gamma": sd[p + "norm1.weight"].float(), "beta": sd[p + "norm1.bias"].float(),
            "Wq": W[:D], "Wk": W[D:2 * D], "bq": sd[p + "attn.q_bias"].float()}


def load_decoder(cfg):
    """(n_features, 768) unit decoder directions (multiplier is exp(0) = 1, asserted)."""
    sd = torch.load(cfg["sae_path"], map_location="cpu", weights_only=True)["sae_state_dict"]
    assert float(sd["dictionary.multiplier"]) == 0.0
    return sd["dictionary._weights"].float()


def project(cfg, dec, blk):
    """Per-head query / key vectors of every feature, (F, H, d_head), and the default pull (H, F)."""
    d = blk["gamma"] * (dec - dec.mean(dim=1, keepdim=True))    # LayerNorm-folded directions
    H = cfg["n_heads"]
    q = (d @ blk["Wq"].T).reshape(len(d), H, -1)
    k = (d @ blk["Wk"].T).reshape(len(d), H, -1)
    u = (blk["beta"] @ blk["Wq"].T + blk["bq"]).reshape(H, -1)   # constant query part
    pull = torch.einsum("fhd,hd->hf", k, u)
    return q, k, pull


def null_stats(cfg, q, k):
    """Per-head mean / std of S over random feature pairs, shape (H,) each."""
    g = torch.Generator().manual_seed(cfg["seed"])
    f1 = torch.randint(len(q), (cfg["n_null"],), generator=g)
    f2 = torch.randint(len(q), (cfg["n_null"],), generator=g)
    s = (q[f1] * k[f2]).sum(-1)                                  # (n_null, H)
    return s.mean(0), s.std(0)


def ramp_table(q, k, ramps, mu, sd):
    """Long table: ramp, head, feature, seek_z (f looks for ramp), sought_z (ramp looks for f)."""
    rows = []
    for r in ramps:
        seek = (torch.einsum("fhd,hd->hf", q, k[r]) - mu[:, None]) / sd[:, None]
        sought = (torch.einsum("hd,fhd->hf", q[r], k) - mu[:, None]) / sd[:, None]
        H, F = seek.shape
        rows.append(pd.DataFrame({
            "ramp": r, "head": torch.arange(H).repeat_interleave(F).numpy(),
            "feature": torch.arange(F).repeat(H).numpy(),
            "seek_z": seek.flatten().numpy(), "sought_z": sought.flatten().numpy()}))
    return pd.concat(rows, ignore_index=True)


def pull_table(pull):
    """Default pull per head, z-scored across features within the head."""
    z = (pull - pull.mean(1, keepdim=True)) / pull.std(1, keepdim=True)
    H, F = z.shape
    return pd.DataFrame({"head": torch.arange(H).repeat_interleave(F).numpy(),
                         "feature": torch.arange(F).repeat(H).numpy(), "pull_z": z.flatten().numpy()})


def head_spread(tab, col):
    """Head x ramp grid of the 99th percentile |z| over features (ramp itself excluded)."""
    t = tab[tab.feature != tab.ramp]
    return t.groupby(["head", "ramp"])[col].apply(lambda s: s.abs().quantile(0.99)).unstack().round(1)


def main(cfg):
    ramps = pd.read_csv(cfg["sets_csv"]).query("arm == 'ramp'").feature.tolist()
    q, k, pull = project(cfg, load_decoder(cfg), load_block(cfg))
    mu, sd = null_stats(cfg, q, k)
    tab, pt = ramp_table(q, k, ramps, mu, sd), pull_table(pull)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    tab.to_parquet(cfg["out_dir"] / f"ramp_pairs_L{cfg['sae_layer']}.parquet")
    pt.to_parquet(cfg["out_dir"] / f"pull_L{cfg['sae_layer']}.parquet")
    for col in ("seek_z", "sought_z"):
        print(f"\n{col}: 99th pct |z| over features, head x ramp\n{head_spread(tab, col)}")


if __name__ == "__main__":
    main(CFG)
