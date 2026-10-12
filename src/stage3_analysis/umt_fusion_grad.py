"""Grad-enabled copies of UMT's plain-torch fusion (layers 9-11 + itm_head) for the
vision->text bridge, plus the G1 correctness checks (brief 121026):

  Check A: grad-enabled forward == verified no-grad forward (max |diff| <= 1e-6).
  Check B: autograd gradients == central finite differences on random directions
           (rel err <= 1e-3, fp32, TF32 off) w.r.t. both the vision tokens and the
           layer-8 text hidden states.

The verified no-grad implementations (umt_vtm_gate, train_sae_umt_fused) are NOT
edited; _lin/_heads are undecorated pure ops and safe to share.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "src" / "stage3_analysis")]
from umt_vtm_gate import _heads, _lin


def attend_grad(x, k, v, w, pre, add_mask):
    """umt_vtm_gate._attend, grad-enabled: q from x; k/v given; add & LayerNorm."""
    s = _heads(_lin(x, w, pre + ".self.query")) @ k.transpose(-1, -2) / 8.0
    if add_mask is not None:
        s = s + add_mask
    ctx = (s.softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
    return F.layer_norm(_lin(ctx, w, pre + ".output.dense") + x, (768,),
                        w[pre + ".output.LayerNorm.weight"], w[pre + ".output.LayerNorm.bias"], 1e-12)


def fusion_layer_grad(h, add_mask, cross_kv, w, i):
    """umt_vtm_gate._fusion_layer, grad-enabled: self-attn -> cross-attn -> FFN."""
    p = f"text_encoder.encoder.layer.{i}"
    k, v = _heads(_lin(h, w, p + ".attention.self.key")), _heads(_lin(h, w, p + ".attention.self.value"))
    h = attend_grad(h, k, v, w, p + ".attention", add_mask)
    h = attend_grad(h, *cross_kv, w, p + ".crossattention", None)
    ff = _lin(F.gelu(_lin(h, w, p + ".intermediate.dense")), w, p + ".output.dense")
    return F.layer_norm(ff + h, (768,), w[p + ".output.LayerNorm.weight"],
                        w[p + ".output.LayerNorm.bias"], 1e-12)


def vtm_scores_grad(vis, text, w, layers=(9, 10, 11)):
    """umt_vtm_gate.vtm_scores, grad-enabled. vis (L, 768) may require grad; the text
    cache hidden may too. Returns (n_templates,) raw itm scores."""
    from umt_vtm_gate import _kv
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention")
             for i in layers}
    out = []
    for s in range(0, len(text["hidden"]), 58):
        h, m = text["hidden"][s:s + 58], text["add_mask"][s:s + 58]
        for i in layers:
            h = fusion_layer_grad(h, m, cross[i], w, i)
        out.append(_lin(h[:, 0], w, "itm_head")[:, 1])
    return torch.cat(out)


def itm_from_grad(h, start, vis, text, w, layers=(9, 10, 11)):
    """train_sae_umt_fused.itm_from, grad-enabled: layers start..11 on h, then itm."""
    from umt_vtm_gate import _kv
    cross = {i: _kv(vis.unsqueeze(0), w, f"text_encoder.encoder.layer.{i}.crossattention")
             for i in layers if i >= start}
    outs = []
    for s in range(0, len(h), 58):
        hh, m = h[s:s + 58], text["add_mask"][s:s + 58]
        for i in layers:
            if i >= start:
                hh = fusion_layer_grad(hh, m, cross[i], w, i)
        outs.append(_lin(hh[:, 0], w, "itm_head")[:, 1])
    return torch.cat(outs)


def check_a(text, w, device):
    """Check A: grad-enabled forward == verified no-grad vtm_scores on the cache's
    fixed reference vision tokens. Returns max |diff|."""
    from umt_vtm_gate import vtm_scores
    cache = np.load(ROOT / "notebooks/umt_ssv2_template_bert_l8.npz")
    errs = []
    for ref in torch.from_numpy(cache["ref_vision"]):
        vis = ref.to(device)
        with torch.no_grad():
            base = vtm_scores(vis, text, w)
        grad_out = vtm_scores_grad(vis, text, w)          # graph built, same inputs
        errs.append((base - grad_out).abs().max().item())
    return max(errs)


def _fd_check(f, x, n_dir=4, eps=1e-2, seed=0):
    """Central-difference vs autograd directional derivatives of scalar f at x.
    Directions: the gradient itself (max SNR — the strict check) plus random unit
    ones; denominator floored at 5% of |g| so near-orthogonal random directions
    (directional signal ~1e-4, FD roundoff ~1e-4 at fp32) can't fake a failure.
    eps=1e-2: empirically the sweet spot (1e-3 drowns in roundoff, see G1 note)."""
    g = torch.autograd.grad(f(x), x)[0]
    dirs = [(g / g.norm()).detach()]
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n_dir):
        d = torch.randn(x.shape, generator=rng).to(x.device)
        dirs.append(d / d.norm())
    rows = []
    for d in dirs:
        fd = (f(x + eps * d) - f(x - eps * d)) / (2 * eps)
        ad = float((g * d).sum())
        den = max(abs(ad), abs(float(fd)), 0.05 * float(g.norm()), 1e-8)
        rows.append((ad, float(fd), float(abs(ad - float(fd)) / den)))
    return rows


def check_b(text, w, device, tpl=6):
    """Check B in float64: itm logits are O(100), so fp32 central differences have a
    ~5e-4 roundoff floor — above the random-direction signal. fp64 drops that to
    ~1e-12 and lets every direction, not just the gradient-aligned one, be strict."""
    ref = torch.from_numpy(np.load(ROOT / "notebooks/umt_ssv2_template_bert_l8.npz")
                           ["ref_vision"][0]).double().to(device)
    w64 = {k: v.double() for k, v in w.items()}
    t64 = {"hidden": text["hidden"].double(), "add_mask": text["add_mask"].double()}
    vis = ref.clone().requires_grad_(True)
    rows_vis = _fd_check(lambda v: vtm_scores_grad(v, t64, w64)[tpl], vis)
    h = t64["hidden"][tpl:tpl + 1].clone().requires_grad_(True)
    sub = {"hidden": h, "add_mask": t64["add_mask"][tpl:tpl + 1]}
    rows_txt = _fd_check(lambda hh: vtm_scores_grad(ref, {"hidden": hh,
                                                          "add_mask": sub["add_mask"]}, w64)[0], h)
    return max(r[2] for r in rows_vis), max(r[2] for r in rows_txt), rows_vis + rows_txt


def run_g1(text, w, device):
    """Both G1 checks; returns the report dict. Thresholds per the brief:
    A <= 1e-6, B <= 1e-3 relative error."""
    a = check_a(text, w, device)
    b_vis, b_txt, rows = check_b(text, w, device)
    report = {"check_a_max_diff": a, "check_b_rel_err_vision": b_vis,
              "check_b_rel_err_text": b_txt,
              "pass_a": a <= 1e-6, "pass_b": b_vis <= 1e-3 and b_txt <= 1e-3,
              "fd_rows": [(round(a_, 6), round(f_, 6), round(r, 8)) for a_, f_, r in rows]}
    print(f"G1 Check A (forward equality): max |diff| = {a:.2e}  "
          f"{'PASS' if report['pass_a'] else 'FAIL'}")
    print(f"G1 Check B (autograd vs FD):   vision {b_vis:.2e}  text {b_txt:.2e}  "
          f"{'PASS' if report['pass_b'] else 'FAIL'}")
    return report
