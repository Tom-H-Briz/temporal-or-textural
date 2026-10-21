"""Overview figure for one L9 SAE feature in the UMT MLM runs (CPU, local).

Top: mean activation per frame position, class A vs B, under R / C (shuffle) / Rev.
Bottom: mean activation per class on R (all MLM classes); per clip activation vs MLM margin T_ab, R and C.
Activation is scaled by the median R activation on REF_CLASS clips ("1 = a typical full-on clip").
Needs: umt_mlm_feature_profile.py output.
Usage: FEATURE=4250 PAIR=45,43 REF_CLASS=43 uv run python src/stage3_analysis/visualisation/umt_mlm_feature_overview.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CFG = {
    "feature":   int(os.environ.get("FEATURE", "4250")),
    "pair":      tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(",")),
    "ref_class": int(os.environ.get("REF_CLASS", "43")),
    "labels":    ROOT / "data/ssv2/labels/labels.json",
    "colours":   ("#eb6834", "#2a78d6"),          # class A, class B (validated categorical slots 2, 1)
    "ink": "#0b0b0b", "muted": "#8a8984", "grid": "#e6e5e0", "surface": "#fcfcfb",
    "conds":     {"R": "Real (R)", "C": "Shuffled (C)", "Rev": "Reversed (Rev)"},
}
CFG["dir"] = ROOT / "outputs/analysis/umt_mlm_feature_profile" / f"f{CFG['feature']}"


def load(cfg):
    """Profile table, class names, and the activation scale (median R z on ref-class clips)."""
    t = pd.read_parquet(cfg["dir"] / "clip_profile.parquet")
    names = {int(v): k for k, v in json.load(open(cfg["labels"])).items()}
    ref = t[(t.true_class == cfg["ref_class"]) & (t.condition == "R")].z.median()
    return t, names, ref


def panel_profiles(fig, gs, pt, names, ref, cfg):
    """Row 1: mean per-frame activation of class A vs B under each condition, shared y."""
    first = None
    for i, (cond, title) in enumerate(cfg["conds"].items()):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2], sharey=first)
        for cls, col in zip(cfg["pair"], cfg["colours"]):
            P = np.stack(pt[(pt.condition == cond) & (pt.true_class == cls)].z_pos) / (ref / 12)
            ax.plot(range(12), P.mean(0), color=col, lw=2, marker="o", ms=4, label=names[cls])
        ax.set_title(title, loc="left", fontsize=10, color=cfg["ink"])
        ax.set_xlabel("frame position (0–11)")
        ax.grid(axis="y", color=cfg["grid"], lw=0.6); ax.set_axisbelow(True)
        if first is None:
            first = ax
            ax.set_ylabel(f"#{cfg['feature']} activation per frame\n(× median real clip, class {cfg['ref_class']})")
            ax.legend(frameon=False, loc="upper left", fontsize=8)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)


def panel_classes(ax, t, names, ref, cfg):
    """Mean R activation per MLM class; the reference class highlighted, values labelled."""
    m = (t[t.condition == "R"].groupby("true_class").z.mean() / ref).sort_values()
    ax.barh([names[c][:38] for c in m.index], m.values, height=0.7,
            color=[cfg["colours"][1] if c == cfg["ref_class"] else "#b5b4ad" for c in m.index])
    for y, v in enumerate(m.values):
        ax.text(v + 0.01 * m.max(), y, f"{v:.2f}", va="center", fontsize=7.5, color="#52514e")
    ax.set_xlabel("mean activation, real video\n(× median real clip, reference class)")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_title("Which classes it fires on (R)", loc="left", fontsize=10, color=cfg["ink"])


def panel_scatter(axes, pt, names, ref, cfg):
    """Per clip activation vs the MLM margin T_ab (> 0 = says word A), R and C, shared x."""
    xmax = 1.05 * pt[pt.condition.isin(["R", "C"])].z.max() / ref
    for ax, cond in zip(axes, ["R", "C"]):
        for cls, col in zip(cfg["pair"][::-1], cfg["colours"][::-1]):
            d = pt[(pt.condition == cond) & (pt.true_class == cls)]
            ax.scatter(d.z / ref, d.T_ab, s=14, color=col, alpha=0.55, lw=0, label=names[cls])
        ax.set_xlim(-0.02 * xmax, xmax); ax.axhline(0, color=cfg["muted"], lw=0.8)
        ax.grid(color=cfg["grid"], lw=0.6)
        ax.set_xlabel(f"#{cfg['feature']} activation (× median real clip)")
        label = cfg["conds"][cond].split(" (")[0].lower()
        ax.set_title(f"MLM decision vs #{cfg['feature']}, {label} ({cond})",
                     loc="left", fontsize=10, color=cfg["ink"])
    axes[0].set_ylabel(f"logit(word A) − logit(word B)\n(> 0 = says A)")
    axes[1].legend(frameon=False, loc="upper right", markerscale=1.5, fontsize=8)


def main():
    cfg = CFG
    t, names, ref = load(cfg)
    pt = t[(t.a == cfg["pair"][0]) & (t.b == cfg["pair"][1])]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": cfg["muted"], "xtick.color": cfg["muted"], "ytick.color": cfg["muted"]})
    fig = plt.figure(figsize=(13, 8.2), facecolor=cfg["surface"])
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.45, wspace=0.9)
    panel_profiles(fig, gs, pt, names, ref, cfg)
    panel_classes(fig.add_subplot(gs[1, 0:2]), t, names, ref, cfg)
    panel_scatter([fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])], pt, names, ref, cfg)
    fig.suptitle(f"UMT L9 #{cfg['feature']}: {names[cfg['pair'][0]]} vs {names[cfg['pair'][1]]} (MLM runs)",
                 x=0.01, ha="left", fontsize=11, color=cfg["ink"])
    out = cfg["dir"] / f"overview_{cfg['pair'][0]}_{cfg['pair'][1]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=cfg["surface"])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
