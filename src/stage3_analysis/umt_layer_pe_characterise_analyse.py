"""Analyse + plot the dictionary-wide PE characterisation (19/11/26; umt_layer_pe_characterise.py). CPU, local.
Per latent and layer: real level (mean R clip), blank / real, best synthetic, class (sheet rule: content if blank < 1%
real; PE driven if blank >= 0.8 x real (pe_cut) and best synthetic < 2x blank; else mixed), frozen / full on blank (time dependence),
fire rate and order index (Rev profile locked vs flipped) on real clips, fixed-threshold all-174 DFA mass and class breadth (n50).
Out: outputs/analysis/umt_layer_pe_characterise/{features_L{L}.csv, summary.csv, plots/*.png}.
Usage: uv run python src/stage3_analysis/umt_layer_pe_characterise_analyse.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from umt_layer_specificity import CFG as SPEC, class_mean, load

D = Path(__file__).resolve().parents[2] / "outputs/analysis/umt_layer_pe_characterise"
# pe_cut 0.8 (20/11/26): pure PE = blank >= 0.8 x real, so VideoMAE's spikes (0.79-0.95 of real) count as pure PE
CFG = {"layers": (5, 7, 9), "content_cut": 0.01, "pe_cut": 0.8, "synth_x": 2.0, "dir": D, "plots": D / "plots",
       "classes": ("pe", "mixed", "content"), "colors": {"pe": "#2a78d6", "mixed": "#eb6834", "content": "#1baf7a"},
       "active": 0.5, "layer_colors": {5: "#2a78d6", 7: "#eb6834", 9: "#1baf7a"}, "bins": np.linspace(-3, 1, 9)}


def frame_corr(a, b):
    """Per clip x latent Pearson r over frames of a, b (clips, 12, F); nan where either profile is constant."""
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def order_index(R, Rev):
    """Median over clips of r(Rev, R) - r(Rev, R reversed): + = profile locked to the PE under a reversed video,
    - = profile follows the content (19/11/26; clip totals carry no order signal, so shape is used).
    20/11/26: a clip-mean version was tried and dropped (> 1 for every class: averaging cancels content timing and
    leaves the position-locked baseline). Per-clip reads ~0 for shallow noisy profiles (VM pure PE)."""
    return np.nanmedian(frame_corr(Rev, R) - frame_corr(Rev, R[:, ::-1]), 0)


def measures(L, cfg):
    """One row per latent at layer L (ordinary tokens; real clips from Isambard, synthetic computed locally)."""
    z, sy = np.load(cfg["dir"] / f"L{L}.npz"), np.load(cfg["dir"] / f"synth_L{L}.npz")
    R, Rev = z["R"].astype(np.float32), z["Rev"].astype(np.float32)                   # (clips, 12, F)
    blank, frozen = (z[k][0].astype(np.float64).sum(0) for k in ("blank_full", "blank_frozen"))
    real, synth = R.sum(1).mean(0).astype(np.float64), sy["synth"].astype(np.float64).sum(1).max(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = pd.DataFrame({"latent": np.arange(len(real)), "real": real, "blank": blank, "best_synth": synth,
                          "blank_real": blank / real, "frozen_kept": frozen / blank,
                          "fire_rate": (R.sum(1) > 0).mean(0), "order_index": order_index(R, Rev)})
    d["cls"] = np.where(d.blank < cfg["content_cut"] * d.real, "content",
                        np.where((d.blank >= cfg["pe_cut"] * d.real) & (d.best_synth < cfg["synth_x"] * d.blank), "pe", "mixed"))
    d.loc[(d.real == 0) & (d.blank == 0), "cls"] = "dead"
    return d.assign(layer=L)


def add_dfa(tabs, cfg):
    """Fixed-threshold all-174 DFA per latent: total class-mean |DFA| (use) and classes for half its mass (n50)."""
    X, cls = load(dict(SPEC, layers=cfg["layers"]))
    for L, d in tabs.items():
        M = class_mean(X[L], cls)
        P = M / np.where(M.sum(0) > 0, M.sum(0), 1)
        d["dfa_mass"] = M.sum(0)
        d["n50"] = np.where(M.sum(0) > 0, (np.cumsum(-np.sort(-P, 0), 0) < 0.5).sum(0) + 1, np.nan)
    return tabs


def summary(tabs, cfg):
    """Per layer and class: count, share of live latents, share of DFA mass, medians of the other measures."""
    rows = []
    for L, d in tabs.items():
        live = d[d.cls != "dead"]
        for c in cfg["classes"]:
            g = live[live.cls == c]
            rows.append({"layer": L, "cls": c, "n": len(g), "pct_latents": 100 * len(g) / len(live),
                         "pct_dfa_mass": 100 * g.dfa_mass.sum() / live.dfa_mass.sum(), "n50": g.n50.median(),
                         "fire_rate": g.fire_rate.median(), "order_index": g.order_index.median(),
                         "frozen_kept": g.frozen_kept.median()})
    return pd.DataFrame(rows)


def summary_active(tabs, cfg):
    """Class shares among active latents only (fire on >= cfg['active'] of real clips)."""
    rows = []
    for L, d in tabs.items():
        a = d[(d.cls != "dead") & (d.fire_rate >= cfg["active"])]
        for c in cfg["classes"]:
            g = a[a.cls == c]
            rows.append({"layer": L, "cls": c, "n_active": len(g), "pct_active": 100 * len(g) / len(a),
                         "pct_dfa_mass_active": 100 * g.dfa_mass.sum() / a.dfa_mass.sum(),
                         "pct_act_mass_active": 100 * g.real.sum() / a.real.sum(),   # 20/11: activation mass
                         "order_index": g.order_index.median(), "n50": g.n50.median()})
    return pd.DataFrame(rows)


def style(ax, title, ylabel):
    """Recessive axes: no top / right spines, light y grid, muted ticks."""
    ax.set_title(title, fontsize=11, loc="left", color="#0b0b0b")
    ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_composition(act, cfg):
    """Stacked bars per layer: class share of active latents (left) and of their DFA mass (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, col, t in zip(axes, ("pct_active", "pct_dfa_mass_active"), ("Share of active latents", "Share of DFA mass")):
        base = np.zeros(len(cfg["layers"]))
        for c in cfg["classes"]:
            v = act[act.cls == c].set_index("layer").loc[list(cfg["layers"]), col].to_numpy()
            ax.bar([f"L{L}" for L in cfg["layers"]], v, bottom=base, color=cfg["colors"][c], width=0.6,
                   edgecolor="white", linewidth=2, label={"pe": "pure PE", "mixed": "PE x content", "content": "content"}[c])
            for i, (b, h) in enumerate(zip(base, v)):
                ax.text(i, b + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=8, color="white")
            base += v
        style(ax, t, "%")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(cfg["plots"] / "composition_by_layer.png", dpi=200); plt.close(fig)


def plot_class_trends(tabs, cfg):
    """Median (line) and IQR (band) per class across layers, active latents: class breadth n50 and order index."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    names = {"pe": "pure PE", "mixed": "PE x content", "content": "content"}
    for ax, col, t in zip(axes, ("n50", "order_index"), ("Classes holding half the DFA mass", "Order index (+ locked to PE)")):
        for c in cfg["classes"]:
            q = np.array([tabs[L].query(f"cls == '{c}' and fire_rate >= {cfg['active']}")[col].quantile([.25, .5, .75])
                          for L in cfg["layers"]])
            ax.fill_between(cfg["layers"], q[:, 0], q[:, 2], color=cfg["colors"][c], alpha=0.15, linewidth=0)
            ax.plot(cfg["layers"], q[:, 1], color=cfg["colors"][c], linewidth=2, marker="o", markersize=6, label=names[c])
            if col == "n50":                                           # order panel: lines meet at 0, so legend
                ax.annotate(names[c], (cfg["layers"][-1], q[-1, 1]), xytext=(6, 0), textcoords="offset points", fontsize=8, va="center")
        ax.set_xticks(cfg["layers"], [f"L{L}" for L in cfg["layers"]]); style(ax, t, "")
    axes[1].axhline(0, color="#c3c2b7", linewidth=1)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(cfg["plots"] / "class_trends.png", dpi=200); plt.close(fig)


def plot_spectrum(tabs, cfg):
    """Left: DFA-mass-weighted histogram of log10 blank / real (active latents; content piled at the -3 floor).
    Right: median n50 per log blank / real bin. One line per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    b = cfg["bins"]; mid = (b[1:] + b[:-1]) / 2
    for L in cfg["layers"]:
        a = tabs[L][(tabs[L].cls != "dead") & (tabs[L].fire_rate >= cfg["active"])]
        x = np.clip(np.log10(a.blank_real.clip(lower=1e-9)), b[0], b[-1] - 1e-9)
        h, _ = np.histogram(x, b, weights=a.dfa_mass); col = cfg["layer_colors"][L]
        axes[0].plot(mid, 100 * h / h.sum(), color=col, linewidth=2, marker="o", markersize=6, label=f"L{L}")
        med = [a.n50[(x >= lo) & (x < hi)].median() for lo, hi in zip(b[:-1], b[1:])]
        axes[1].plot(mid, med, color=col, linewidth=2, marker="o", markersize=6, label=f"L{L}")
    style(axes[0], "Where the DFA mass sits on the PE spectrum", "% of DFA mass")
    style(axes[1], "Class breadth vs PE drive", "median classes for half the mass")
    for ax in axes:
        ax.invert_xaxis()                                      # 20/11/26: pure PE on the left in every spectrum plot
        ax.set_xlabel("log10 blank / real  (left: PE driven, right: content)", fontsize=8, color="#52514e")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(cfg["plots"] / "pe_spectrum.png", dpi=200); plt.close(fig)


def correlations(tabs, cfg):
    """Spearman, active latents per layer: log10 blank / real vs class breadth (n50) and vs order index."""
    rows = []
    for L, d in tabs.items():
        a = d[(d.cls != "dead") & (d.fire_rate >= cfg["active"])].dropna(subset=["n50", "order_index"])
        x = np.log10(a.blank_real.clip(lower=1e-9))
        for col in ("n50", "order_index"):
            r, p = spearmanr(x, a[col]); rows.append({"layer": L, "vs": col, "rho": r, "p": p, "n": len(a)})
    return pd.DataFrame(rows)


def main(cfg):
    cfg["plots"].mkdir(parents=True, exist_ok=True)
    tabs = add_dfa({L: measures(L, cfg) for L in cfg["layers"]}, cfg)
    for L, d in tabs.items():
        d.to_csv(cfg["dir"] / f"features_L{L}.csv", index=False)
    s, act, cor = summary(tabs, cfg), summary_active(tabs, cfg), correlations(tabs, cfg)
    for n, t in (("summary", s), ("summary_active", act), ("correlations", cor)):
        t.round(4).to_csv(cfg["dir"] / f"{n}.csv", index=False); print(t.round(3).to_string(index=False), "\n")
    plot_composition(act, cfg); plot_class_trends(tabs, cfg); plot_spectrum(tabs, cfg)


if __name__ == "__main__":
    main(CFG)
