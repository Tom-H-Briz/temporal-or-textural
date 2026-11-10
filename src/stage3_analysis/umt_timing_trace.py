"""
Which downstream SAE latents are built from the timing features? Three CPU reads, SRC -> DST layer:
  A direct : DST encoder applied to SRC timing decoder directions (residual skip path)
  B OV     : DST encoder applied to each in-between block's per-head OV image of the timing directions
  C covary : within-class residual correlation of SRC timing and DST latent frame profiles (same 1,400 clips)
A / B are z-scored per DST latent against all (A) or 1,024 random (B) SRC features. Weights / correlational only.
Usage: SRC_LAYER=5 DST_LAYER=9 uv run python src/stage3_analysis/umt_timing_trace.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
S, D = int(os.environ.get("SRC_LAYER", 5)), int(os.environ.get("DST_LAYER", 9))
CFG = {
    "src": S, "dst": D, "n_heads": 12, "n_null_ov": 1024, "seed": 0, "show": 15, "targets": [393, 4250] if D == 9 else [],   # L9 gated up / down ramps
    "sae": ROOT / "outputs/sae/sae_umt_ssv2_k64_x8_l{L}_job7ep_best.pt",
    "timing_csv": ROOT / f"outputs/analysis/umt_layer_positional/timing_features_L{S}.csv",
    "dst_timing_csv": ROOT / f"outputs/analysis/umt_layer_positional/timing_features_L{D}.csv",
    "pairs_csv": ROOT / f"outputs/analysis/umt_pair_contrast_L{D}/features.csv",
    "prof": ROOT / "outputs/analysis/umt_layer_positional_ablation_L{L}_splice/profiles_R.npz",
    "hf_repo": "OpenGVLab/UMT", "hf_file": "multi_modality/ret_ssv2_tpl_b16_25m.pth",
    "out_dir": ROOT / "outputs/analysis/umt_timing_trace",
}


def sae(cfg, L):
    """Unit decoder (F, 768), encoder weight (F, 768) for layer L."""
    sd = torch.load(str(cfg["sae"]).format(L=L), map_location="cpu", weights_only=True)["sae_state_dict"]
    return sd["dictionary._weights"].float(), sd["encoder.final_block.0.weight"].float()


def ov_blocks(cfg):
    """For each block between the SRC hook and DST hook: (block, γ, W_V (768, 768), W_O (768, 768))."""
    from huggingface_hub import hf_hub_download
    sd = torch.load(hf_hub_download(cfg["hf_repo"], cfg["hf_file"]), map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    out = []
    for b in range(cfg["src"] + 1, cfg["dst"] + 1):
        p = f"vision_encoder.encoder.blocks.{b}."
        W = sd[p + "attn.qkv.weight"].float()
        out.append((b, sd[p + "norm1.weight"].float(), W[1536:], sd[p + "attn.proj.weight"].float()))
    return out


def read_z(dirs, null, W_enc):
    """(T, F_dst) z of each DST latent's encoder read of each direction, vs its read of the null directions."""
    r, n = dirs @ W_enc.T, null @ W_enc.T
    return ((r - n.mean(0)) / n.std(0)).numpy()


def ov_image(dirs, gamma, Wv, Wo, h, d_head=64):
    """Head h's OV output for LayerNorm-folded directions (n, 768)."""
    s = slice(h * d_head, (h + 1) * d_head)
    x = gamma * (dirs - dirs.mean(1, keepdim=True))
    return x @ Wv[s].T @ Wo[:, s].T


def path_b(cfg, dec, timing, W_enc):
    """Strongest OV read over (block, head): signed z (T, F) and its 'b{block}h{head}' label."""
    g = torch.Generator().manual_seed(cfg["seed"])
    null = dec[torch.randperm(len(dec), generator=g)[:cfg["n_null_ov"]]]
    zs, labs = [], []
    for b, gamma, Wv, Wo in ov_blocks(cfg):
        for h in range(cfg["n_heads"]):
            zs.append(read_z(ov_image(dec[timing], gamma, Wv, Wo, h), ov_image(null, gamma, Wv, Wo, h), W_enc))
            labs.append(f"b{b}h{h}")
    Z = np.stack(zs)                                           # (n_bh, T, F)
    i = np.abs(Z).argmax(0)
    return np.take_along_axis(Z, i[None], 0)[0], np.array(labs)[i]


def class_residual(cfg, L, cols=None):
    """(clip x frame, k) profiles minus their class-mean frame profile; plus clip ids."""
    with np.load(str(cfg["prof"]).format(L=L)) as z:
        ids, cls, p = z["clip_id"], z["class_id"], z["prof"].astype(np.float32)
    p = np.concatenate([p if cols is None else p[:, :, cols], p.sum(2, keepdims=True)], 2)  # last col = total
    for c in np.unique(cls):
        p[cls == c] -= p[cls == c].mean(0)
    p = p.reshape(-1, p.shape[-1])
    return ids, p[:, :-1], p[:, -1]


def partial_out(X, covs):
    """Residualise columns of X on the covariate columns (least squares)."""
    beta, *_ = np.linalg.lstsq(covs, X, rcond=None)
    return X - covs @ beta


def path_c(cfg, timing):
    """(T, F_dst) correlation of within-class residual frame profiles, SRC timing vs every DST latent."""
    ids_s, X, ms = class_residual(cfg, cfg["src"], timing)
    ids_d, Y, md = class_residual(cfg, cfg["dst"])
    assert (ids_s == ids_d).all(), "SRC / DST profile clip order differs"
    covs = np.stack([np.ones_like(ms), ms, md], 1)             # total mass at both layers (comb swing)
    X, Y = partial_out(X, covs), partial_out(Y, covs)
    return (X.T @ Y) / np.sqrt(np.outer((X ** 2).sum(0), (Y ** 2).sum(0)) + 1e-12)


def strongest(Z, timing, prefix):
    """Per DST latent: signed score at max |score| over timing features, and which one."""
    i = np.abs(Z).argmax(0)
    return {f"{prefix}": Z[i, np.arange(Z.shape[1])], f"{prefix}_with": np.array(timing)[i]}


def dst_labels(cfg):
    """DST latent -> 'timing:<group>' or 'content:<pair dirs>'."""
    t = pd.read_csv(cfg["dst_timing_csv"]).set_index("feature").group
    p = pd.read_csv(cfg["pairs_csv"]).groupby("latent").home.apply(lambda s: ",".join(map(str, sorted(set(s)))))
    lab = {f: f"content:{h}" for f, h in p.items()}
    lab.update({f: f"timing:{g}" for f, g in t.items()})
    return lab


def main(cfg):
    timing = pd.read_csv(cfg["timing_csv"]).feature.astype(int).tolist()
    dec, _ = sae(cfg, cfg["src"])
    _, W_enc = sae(cfg, cfg["dst"])
    t = pd.DataFrame({"feature": np.arange(W_enc.shape[0])})
    t = t.assign(**strongest(read_z(dec[timing], dec, W_enc), timing, "A_z"))
    zb, lb = path_b(cfg, dec, timing, W_enc)
    t = t.assign(**strongest(zb, timing, "B_z"), B_head=lb[np.abs(zb).argmax(0), np.arange(zb.shape[1])])
    t = t.assign(**strongest(path_c(cfg, timing), timing, "C_r"))
    t["label"] = t.feature.map(dst_labels(cfg)).fillna("")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    t.round(3).to_csv(cfg["out_dir"] / f"trace_L{cfg['src']}_to_L{cfg['dst']}.csv", index=False)
    report(cfg, t)


def report(cfg, t):
    """Top latents per path, label mix and cross-path overlap in the top 50, target rows with percentiles."""
    tops = {}
    for col in ("A_z", "B_z", "C_r"):
        t[f"{col}_pct"] = t[col].abs().rank(pct=True) * 100
        top = t.reindex(t[col].abs().sort_values(ascending=False).index)
        tops[col] = set(top.feature.head(50))
        mix = top.head(50).label.str.split(":").str[0].replace("", "other").value_counts().to_dict()
        print(f"\n=== L{cfg['src']} -> L{cfg['dst']}  path {col[0]}  top-50 mix {mix}")
        print(top.head(cfg["show"]).drop(columns=[c for c in t if c.endswith("_pct")]).round(2).to_string(index=False))
    for a, b in (("A_z", "B_z"), ("A_z", "C_r"), ("B_z", "C_r")):
        print(f"top-50 overlap {a[0]}-{b[0]}: {len(tops[a] & tops[b])}")
    if cfg["targets"]:
        print("\ntargets\n" + t[t.feature.isin(cfg["targets"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
