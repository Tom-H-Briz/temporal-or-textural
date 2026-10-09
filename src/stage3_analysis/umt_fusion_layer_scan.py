"""UMT fusion layer scan: null video x verb pairs -> which fusion layer carries the verb signal?

Gate test for the intervention layer. Content-free video (white / black / iid noise)
removes visual evidence; every cached template runs through the fusion layers
(text-encoder 9-11, cross-attention onto vision tokens) and each minimal verb pair
is measured per layer: CLS/token state delta + cosine, cross-attention readout TV
distance over the 2,352 vision tokens, per-frame attention-mass swing, itm margin.
Layer 8 (pre-fusion, cached) is the text-only control. Pair metrics are percentiled
against the all-15,051-pair template background.
The vision tower never receives text, so verb changes cannot appear in vision-tower
activations (incl. the L5/7/9 SAEs); the only text->video channel is the cross-attention
readout in layers 9-11. The scan targets the fused text states + attention maps.
Outputs (outputs/analysis/umt_fusion_layer_scan/): layer_scan_pairs.csv,
layer_scan_summary.csv + stdout table.
Usage: uv run --with easydict --with "timm==0.4.12" python
    src/stage3_analysis/umt_fusion_layer_scan.py    (CPU, minutes)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "src" / "stage3_analysis")]
from umt_text_geometry import CFG as GEOM_CFG                      # the 15 contrast pairs
from umt_vtm_gate import _attend, _heads, _kv, _lin, load_vtm, vtm_scores
from umt_wrapper import CFG as UMT_CFG, UMTClassifier, UMTProcessor

CFG = {
    "n_frames":      12,    # UMT input; pos_lock_synthetic_probe.py's 16 is VideoMAE's
    "frame_size":    224,
    "noise_mean":    128,
    "noise_std":     64,
    "n_noise_seeds": 16,    # white/black deterministic -> 18 null videos total
    "layers":        (9, 10, 11),   # fusion layers; 8 comes free from the cache as control
    "tpl_chunk":     58,            # 174 = 3 chunks, same attention-memory bound as vtm_scores
    "out_dir":       ROOT / "outputs/analysis/umt_fusion_layer_scan",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}


def make_frames(content: str, seed: int) -> list[np.ndarray]:
    """CFG["n_frames"] uint8 (H, W, 3) frames of one content type: white 255 / black 0 /
    clipped iid noise (temporally varying, deterministically seeded)."""
    h = w = CFG["frame_size"]
    if content == "white":
        return [np.full((h, w, 3), 255, dtype=np.uint8) for _ in range(CFG["n_frames"])]
    if content == "black":
        return [np.full((h, w, 3), 0, dtype=np.uint8) for _ in range(CFG["n_frames"])]
    rng = np.random.default_rng(seed)
    return [np.clip(rng.normal(CFG["noise_mean"], CFG["noise_std"], (h, w, 3)),
                    0, 255).astype(np.uint8) for _ in range(CFG["n_frames"])]


def synthetic_videos() -> list[tuple[str, int]]:
    """(content, seed) per null video — white/black once each (deterministic), noise x N seeds."""
    return [("white", 0), ("black", 0)] + [("noise", s) for s in range(CFG["n_noise_seeds"])]


def _fusion_layer_attn(h, add_mask, cross_kv, w, i):
    """vtm_gate._fusion_layer with the cross-attention probs exposed (same ops, same order):
    self-attn -> cross-attn -> FFN, post-LN. Cross K/V come from the video only."""
    p = f"text_encoder.encoder.layer.{i}"
    h = _attend(h, *_kv(h, w, p + ".attention"), w, p + ".attention", add_mask)
    q = _heads(_lin(h, w, p + ".crossattention.self.query"))
    probs = (q @ cross_kv[0].transpose(-1, -2) / 8.0).softmax(-1)   # (N, 12, Lq, 2352), no vision mask
    ctx = (probs @ cross_kv[1]).transpose(1, 2).reshape(h.shape)
    h = F.layer_norm(_lin(ctx, w, p + ".crossattention.output.dense") + h, (768,),
                     w[p + ".crossattention.output.LayerNorm.weight"],
                     w[p + ".crossattention.output.LayerNorm.bias"], 1e-12)
    ff = _lin(F.gelu(_lin(h, w, p + ".intermediate.dense")), w, p + ".output.dense")
    return F.layer_norm(ff + h, (768,), w[p + ".output.LayerNorm.weight"],
                        w[p + ".output.LayerNorm.bias"], 1e-12), probs


@torch.no_grad()
def fusion_pass(vis, text, w, real):
    """One clip's vision tokens (2352, 768), all 174 cached templates: per-layer hidden
    states {8,9,10,11}, head+real-query-mean cross-attn maps {9,10,11} (174, 2352) with
    rows summing to 1, and raw itm_head[:, 1] scores (174,)."""
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention")
             for i in CFG["layers"]}
    n = len(text["hidden"])
    states, attn, itm_parts = {8: text["hidden"]}, {i: [] for i in CFG["layers"]}, []
    for s in range(0, n, CFG["tpl_chunk"]):
        h, m = text["hidden"][s:s + CFG["tpl_chunk"]], text["add_mask"][s:s + CFG["tpl_chunk"]]
        qmask, parts = real[s:s + CFG["tpl_chunk"]], {}
        for i in CFG["layers"]:
            h, probs = _fusion_layer_attn(h, m, cross[i], w, i)
            parts[i] = h
            mean_q = (probs * qmask[:, None, :, None]).sum(2) / qmask.sum(1)[:, None, None]
            attn[i].append(mean_q.mean(1))          # (chunk, 2352): mean over heads
        for i in CFG["layers"]:
            states.setdefault(i, []).append(parts[i])
        itm_parts.append(_lin(parts[CFG["layers"][-1]][:, 0], w, "itm_head")[:, 1])
    states = {L: torch.cat(v) if isinstance(v, list) else v for L, v in states.items()}
    attn = {i: torch.cat(a) for i, a in attn.items()}
    return states, attn, torch.cat(itm_parts)


def verify_against_vtm(vis, text, w, itm):
    """The attention-exposed path must reproduce vtm_scores (same maths, no probs)."""
    ref = vtm_scores(vis, text, w)
    err = (ref - itm).abs().max().item()
    print(f"fusion_pass vs vtm_scores self-check: max |diff| = {err:.2e}")
    assert err < 1e-4, "attention-exposed fusion diverges from the checked path"


def _iu(m):
    """Upper-triangle values of a symmetric (N, N) matrix."""
    idx = torch.triu_indices(m.shape[0], m.shape[0], 1)
    return m[idx[0], idx[1]]


def _bg_state(X):
    """Background (cos, rel_delta) over all 15,051 template pairs; X (N, D).
    rel = 2*||a-b|| / (||a|| + ||b||), 0=identical, 2=orthogonal-ish."""
    r = X.norm(dim=1)
    rel = 2 * torch.cdist(X, X) / (r[:, None] + r[None, :])
    cos = F.cosine_similarity(X[:, None, :], X[None, :, :], dim=-1)
    return _iu(cos), _iu(rel)


def _pct(bg, v):
    return float((bg < v).double().mean() * 100)


def _bg_tv(A, chunk=48):
    """Pairwise TV distance 0.5*L1 between attention rows; A (N, 2352), rows sum to 1.
    Broadcast in row blocks to bound the (chunk, N, 2352) temporary."""
    n, vals = len(A), []
    for s in range(0, n, chunk):
        vals.append((A[s:s + chunk, None, :] - A[None, :, :]).abs().sum(-1).mul(0.5))
    return _iu(torch.cat(vals))


def _frame_mass(A):
    """(N, 2352) attention rows -> (N, 12) per-frame mass; tokens are time-major (frame x 196)."""
    return A.view(A.shape[0], CFG["n_frames"], -1).sum(-1)


def _bg_fmax(Fm):
    """Pairwise max |Δ| of per-frame attention mass; Fm (N, 12)."""
    return _iu((Fm[:, None, :] - Fm[None, :, :]).abs().amax(-1))


def pair_rows(states, attn, itm, real, content, seed):
    """One row per (contrast pair, layer): state/attention/itm metrics plus each
    metric's percentile within the all-template-pair background."""
    tok = {L: (states[L] * real[:, :, None]).flatten(1) for L in states}   # padding zeroed
    cls = {L: states[L][:, 0, :] for L in states}
    bg = {L: (_bg_state(cls[L]), _bg_state(tok[L])) for L in states}
    tab = {i: (_bg_tv(attn[i]), _bg_fmax(_frame_mass(attn[i]))) for i in CFG["layers"]}
    Fm = {i: _frame_mass(attn[i]) for i in CFG["layers"]}
    bg_itm = _iu((itm[:, None] - itm[None, :]).abs())
    rows = []
    for L in sorted(states):
        (cos_c, rel_c), _ = bg[L]
        for a, b, name in GEOM_CFG["pairs"]:
            row = dict(content=content, seed=seed, pair=name, a=a, b=b, layer=L,
                       cls_cos=float(F.cosine_similarity(cls[L][a], cls[L][b], dim=0)),
                       cls_rel=float(2 * (cls[L][a] - cls[L][b]).norm()
                                     / (cls[L][a].norm() + cls[L][b].norm())),
                       tok_rel=float(2 * (tok[L][a] - tok[L][b]).norm()
                                     / (tok[L][a].norm() + tok[L][b]).norm()))
            row["pct_cls_rel"] = _pct(rel_c, row["cls_rel"])
            if L in CFG["layers"]:
                tv, _ = tab[L]
                row["attn_tv"] = float(0.5 * (attn[L][a] - attn[L][b]).abs().sum())
                row["frame_maxdiff"] = float((Fm[L][a] - Fm[L][b]).abs().max())
                row["pct_attn_tv"] = _pct(tv, row["attn_tv"])
                if L == CFG["layers"][-1]:
                    row["itm_delta"] = float(itm[a] - itm[b])
                    row["pct_itm_abs"] = _pct(bg_itm, abs(row["itm_delta"]))
            rows.append(row)
    return rows


def main():
    import time
    from huggingface_hub import hf_hub_download
    from umt_vtm_gate import CFG as GATE_CFG, vision_and_vtc
    torch.backends.cuda.matmul.allow_tf32 = False       # same precision guard as the VTM gate
    torch.backends.cudnn.allow_tf32 = False
    out = CFG["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={CFG['device']}  nulls={len(synthetic_videos())}  "
          f"pairs={len(GEOM_CFG['pairs'])}  out={out}")
    model = UMTClassifier.from_pretrained().to(CFG["device"]).eval()
    sd = torch.load(hf_hub_download(UMT_CFG["hf_repo"], UMT_CFG["hf_filename"]),
                    map_location="cpu", weights_only=False)
    text, w = load_vtm(sd, CFG["device"])               # runs the xbert-reference self-check
    real = torch.from_numpy(np.load(GATE_CFG["text_cache"])["mask"]).float().to(CFG["device"])
    rows, t0 = [], time.time()
    for k, (content, seed) in enumerate(synthetic_videos()):
        pv = UMTProcessor()(make_frames(content, seed),
                            return_tensors="pt")["pixel_values"].to(CFG["device"])
        vis = vision_and_vtc(model, pv)[0][0]           # (2352, 768) cross-attention input
        states, attn, itm = fusion_pass(vis, text, w, real)
        if k == 0:
            verify_against_vtm(vis, text, w, itm)
        rows += pair_rows(states, attn, itm, real, content, seed)
        print(f"  [{k + 1}/{len(synthetic_videos())}] {content} seed={seed}  "
              f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "layer_scan_pairs.csv", index=False)
    metric_cols = ["cls_cos", "cls_rel", "tok_rel", "attn_tv", "frame_maxdiff", "itm_delta",
                   "pct_cls_rel", "pct_attn_tv", "pct_itm_abs"]
    summ = (df.groupby(["content", "layer"])[metric_cols].median().round(4).reset_index())
    summ.to_csv(out / "layer_scan_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nMedian over the 15 contrast pairs, per null content and layer "
          "(pct_* = percentile within the all-15,051-pair background):")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
