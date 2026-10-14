"""One clip, the pair's top-5 differential features (UMT Phase 3, text side). Per feature two
rows of the 12 frames: R, and Rev (same frames reversed, what the model saw) underneath. The
overlay is the feature's TOKEN-level contribution to the VTM margin (-AP/2 per token, computed
live; red = toward the clip's true template), so sign structure within the clip is visible.
Header per feature: pair-level ablate / flip impact (delta logit gap, delta accuracy) for the
feature and its mass-matched control (phase3_ablation_impact.py), plus this clip's own deltas.
Usage: uv run --with easydict --with "timm==0.4.12" python src/stage3_analysis/visualisation/phase3_top5_clip.py --pair 45 43 [--clip ID]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis"),
                str(Path(__file__).parent)]

CFG = {   # --model umt = Phase 3 text side (T_vtm); videomae / timesformer = classifier port (T_cls)
    "p3_dir": ROOT / "outputs/analysis/umt_vision_text_bridge/phase3",
    "port_dir": ROOT / "outputs/analysis/vision_pair_phase3",
    "video_dir": ROOT / "data/ssv2_val_set",
    "target": "T_vtm", "n_feat": 5, "n_frames": 12, "grid": 14, "img": 224,
}


def pick_clip(df, feats, tgt):
    """Default clip: a held-out intervention clip the model gets right (baseline margin > 0)
    where the top features are most active (mean token share over the features)."""
    s = df[df.kind.eq("single") & df.dose.eq(2.0) & (df[f"base_{tgt}"] > 0)]   # singles: feats is one id
    s = s[s.feats.astype(int).isin(feats)]
    return str(s.groupby("clip_id").tok_frac.mean().idxmax())


def impact_line(imp, name, f):
    """'ablate Δgap -x (acc -y pt) | flip ...' for feature f from ablation_impact_per_feature."""
    r = imp[(imp.pair == name) & (imp.feats.astype(int) == f)].set_index("dose")
    return " | ".join(f"{lab} Δgap {r.d_margin[d]:+.2f} (acc {100 * r.d_acc[d]:+.1f} pt)"
                      for lab, d in (("ablate", 1.0), ("flip", 2.0)))


def clip_line(df, cid, f, tgt):
    """This clip's own real Δgap for f (ablate / flip), from the intervention records."""
    r = df[(df.clip_id == cid) & df.kind.str.endswith("single")]               # singles: feats is one id
    r = r[r.feats.astype(int) == f].drop_duplicates("dose").set_index("dose")[f"delta_{tgt}"]
    return f"this clip Δgap {r.get(1.0, np.nan):+.2f} / {r.get(2.0, np.nan):+.2f}"


def contrib_maps(setup, pv, sp, ti, feats, cfg):
    """Token-level contribution to T (= -AP/2 per token, raw-residual AP of umt_bridge_phase3)
    for each feature -> (n_feat, 12, 14, 14), plus the baseline margin T for this input."""
    from umt_bridge_phase3 import clip_ap, tower_to_layer
    x9 = tower_to_layer(setup["model"], pv.to(setup["real"].device))
    _, ap_tok, _, t_base = clip_ap(x9, setup, sp, setup["matches"], ti)
    tok = -0.5 * ap_tok[cfg["target"]][:, feats]                 # (2352, n_feat)
    g = cfg["grid"]
    return tok.T.reshape(len(feats), cfg["n_frames"], g, g), t_base[cfg["target"]]


def draw_feature(axes_r, axes_v, frames, m_r, m_v, label):
    """Two rows (R, Rev) for one feature; symmetric RdBu_r scale per feature, shared by its two
    rows (99th pct of |contribution|), red = toward the true template. Rev shows the frames in
    the order the model saw them."""
    import matplotlib.cm as cm
    from feature_vis_vm import _overlay
    vmax = float(np.percentile(np.abs(np.concatenate([m_r.ravel(), m_v.ravel()])), 99)) or 1e-6
    norm = mcolors.Normalize(-vmax, vmax)
    for axes, fr, m, tag in ((axes_r, frames, m_r, "R"), (axes_v, frames[::-1], m_v, "Rev")):
        for t, ax in enumerate(axes):
            ax.imshow(_overlay(fr[t], m[t], norm, cm.RdBu_r))
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].set_ylabel(f"{label}\n{tag}  ±{vmax:.2g}" if tag == "R" else tag,
                           fontsize=7, rotation=0, ha="right", va="center")


def load_inputs(cfg, pa, pb):
    """Top-n selection (+ controls), intervention records with baselines, per-feature impact.
    UMT: one Phase 3 dir; port: per-pair dir, impact table in the run dir."""
    tgt, d, imp_dir = cfg["target"], cfg["data_dir"], cfg["impact_dir"]
    sel = pd.read_csv(d / "pair_selection.csv")
    sel = sel[(sel.a == pa) & (sel.b == pb) & (sel.target == tgt)].sort_values("rank").head(cfg["n_feat"])
    base = pd.read_parquet(d / "per_clip_ap.parquet")
    df = pd.read_parquet(d / "per_clip_interventions.parquet").merge(base[["clip_id", f"base_{tgt}"]], on="clip_id")
    df = df[(df.a == pa) & (df.b == pb) & (df.target_src == tgt)]
    imp = pd.read_csv(imp_dir / "ablation_impact_per_feature.csv")
    return sel, df, imp[imp.target == tgt], base.set_index("clip_id").true_class


def set_model(cfg, model, layer, pa, pb):
    """Paths, target and output dir for the chosen model."""
    if model == "umt":
        cfg.update(data_dir=cfg["p3_dir"], impact_dir=cfg["p3_dir"], target="T_vtm")
    else:
        run = cfg["port_dir"] / f"{model}_L{layer}"
        cfg.update(data_dir=run / f"{pa}_{pb}", impact_dir=run, target="T_cls")
    cfg.update(model=model, layer=layer, out_dir=cfg["impact_dir"] / "figures")
    return cfg


def port_maps(st, pv, t, p, feats, model):
    """Classifier port: token-level contribution grad.z (= -AP/2) of T = logit(t) - logit(p) at
    the raw layer-L residual (vision_pair_phase3's path), grouped by temporal position with
    gather_by_position (VM tubelets, TF frames) -> (n_feat, P, 14, 14), plus baseline T."""
    import torch
    from ToT_utils import gather_by_position
    from vision_pair_phase3 import encode_clip, forward_T
    T, x, z0 = encode_clip(st, pv, t, p)
    z = z0.clone().requires_grad_(True)
    g = torch.autograd.grad(forward_T(st, pv, (x + st["sae"].decode(z - z0)).unsqueeze(0), t, p)[0], z)[0]
    m = gather_by_position((g * z0)[:, feats], model).permute(2, 0, 1)      # (n_feat, P, 196)
    return m.reshape(len(feats), m.shape[1], 14, 14).cpu().numpy(), T


def pv_frames(pv, model):
    """Display frames from the model's own input (1, T, C, H, W), min-max scaled per frame AND
    channel (undoes the processor's per-channel normalisation up to contrast); VM shows the first
    frame of each 2-frame tubelet so frames line up with token positions."""
    x = pv[0].permute(0, 2, 3, 1).cpu().numpy()                       # (T, H, W, C)
    lo, span = x.min((1, 2), keepdims=True), np.ptp(x, axis=(1, 2), keepdims=True) + 1e-8
    x = (255 * (x - lo) / span).astype(np.uint8)
    return list(x[::2] if model == "videomae" else x)


def run_port(cid, cls, pa, pb, feats, cfg):
    """Classifier port: model + layer-L SAE hook, the clip via the port's loader, maps for R and
    Rev, and display frames. (VM Rev also reverses frames inside each tubelet.)"""
    import vision_pair_phase3 as vp
    from umt_vision_pair_contrast import clip_loader
    vcfg = dict(vp.CFG, model=cfg["model"], layer=cfg["layer"], num_workers=0)
    st = vp.setup(vcfg)
    pv, _ = next(iter(clip_loader([(cid, cls)], vcfg, "R", st["processor"])))
    t, p = (pa, pb) if cls == pa else (pb, pa)
    return (port_maps(st, pv, t, p, feats, cfg["model"]), port_maps(st, pv.flip(1), t, p, feats, cfg["model"]),
            pv_frames(pv, cfg["model"]))


def run_model(cid, cls, pa, pb, feats, cfg):
    """Phase 3 setup, the clip's pixels (UMT sampler), contribution maps for R and Rev."""
    from stage2_sae.train_sae_umt_fused import build_loader
    from umt_bridge_phase3 import CFG as P3, phase3_setup
    p3 = dict(P3, batch_size=1, num_workers=0)
    setup = phase3_setup(p3)
    pv, _ = next(iter(build_loader([(cid, cls)], p3, shuffle=False)))
    sp, ti = setup["spans"][f"{pa}_{pb}"], 0 if cls == pa else 1
    return contrib_maps(setup, pv, sp, ti, feats, cfg), contrib_maps(setup, pv.flip(1), sp, ti, feats, cfg)


def main():
    import json
    from phase3_ablation_impact import CFG as ABL
    from umt_pair_feature_tracks import load_frames
    p = argparse.ArgumentParser()
    p.add_argument("--pair", type=int, nargs=2, default=[45, 43])
    p.add_argument("--clip", default=None, help="default: best held-out clip for these features")
    p.add_argument("--model", default="umt", choices=["umt", "videomae", "timesformer"])
    p.add_argument("--layer", type=int, default=9)
    a = p.parse_args()
    (pa, pb), name = a.pair, ABL["pair_names"][tuple(a.pair)]
    cfg = set_model(dict(CFG), a.model, a.layer, pa, pb)
    sel, df, imp, cls_of = load_inputs(cfg, pa, pb)
    feats, ctrls = sel.feature.astype(int).tolist(), sel.control.astype(int).tolist()
    cid = a.clip or pick_clip(df, feats, cfg["target"])
    cls = int(cls_of[cid])
    if a.model == "umt":
        (m_r, t_r), (m_v, t_v) = run_model(cid, cls, pa, pb, feats, cfg)
        frames = load_frames(cid, cfg)
    else:
        (m_r, t_r), (m_v, t_v), frames = run_port(cid, cls, pa, pb, feats, cfg)
    labels = {int(v): k for k, v in json.load(open(ROOT / "data/ssv2/labels/labels.json")).items()}
    n, T = len(feats), m_r.shape[1]
    fig = plt.figure(figsize=(1.33 * T + 1.0, 2.8 * n + 1.0))         # square panels for 8 or 12 positions
    gs = fig.add_gridspec(3 * n, T, height_ratios=[0.32, 1, 1] * n, hspace=0.06, wspace=0.03,
                          left=0.07, right=0.995, top=0.93, bottom=0.01)   # per feature: header, R, Rev
    for i, (f, c) in enumerate(zip(feats, ctrls)):
        typ = "sign-reversed" if sel.ratio.iloc[i] < 0 else "one-sided"
        hax = fig.add_subplot(gs[3 * i, :])
        hax.axis("off")
        hax.text(0, 0, f"#{f} ({typ}, differential rank {i + 1}):  {impact_line(imp, name, f)}   ||   "
                       f"{clip_line(df, cid, f, cfg['target'])}\ncontrol #{c}:  {impact_line(imp, name, c)}",
                 fontsize=7.5, va="bottom")
        draw_feature([fig.add_subplot(gs[3 * i + 1, t]) for t in range(T)],
                     [fig.add_subplot(gs[3 * i + 2, t]) for t in range(T)], frames, m_r[i], m_v[i], f"#{f}")
    what = ("UMT Phase 3 (text side), VTM gap" if a.model == "umt"
            else f"{a.model} L{a.layer} classifier, logit gap")
    fig.suptitle(f"{what} (true − partner), {name}: clip {cid}, true = \"{labels[cls]}\"   |   R {t_r:+.2f}, Rev {t_v:+.2f}\n"
                 f"Overlay: token-level contribution to the gap, red = toward the true class. "
                 f"Δgap / acc = pair mean over held-out clips, single feature (k=1).", fontsize=9)
    out = cfg["out_dir"] / f"top5_{pa}_{pb}_{cid}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
