"""#4250 inputs by layer (20/11/26; uv run python src/stage3_analysis/umt_4250_inputs_by_layer.py, CPU, local).
#4250's top-20 inputs at L5 and L7 (|linear attribution|, 311 down clips), bucketed by the
sheet rule (content: blank < 1% real; pure PE: blank >= real and best synth < 2x blank; else mixed); profiles on down
clips, per-frame attribution, and one PE-spectrum panel with #4250 itself (L9) as the end point."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
OUT = A / "umt_4250_inputs_by_layer"; OUT.mkdir(parents=True, exist_ok=True)
PC = A / "umt_layer_pe_characterise"
DOWN = (pd.read_csv(A / "umt_clock_downstream_L5_raw/clips.csv").class_id == 43).to_numpy()   # same rows in L7 run
POS, NEG = "#2a78d6", "#eb6834"


def attr_by_frame(L):
    """(12, F) per-frame linear attribution of #4250 to layer-L latents, % of #4250's mean clip-summed activation."""
    D = A / f"umt_clock_downstream_L{L}_raw"
    at, tg = (np.load(D / f"{n}_R.npy", mmap_mode="r") for n in ("attr", "attr_tg"))
    prof = np.load(D / "prof_L9_R.npy", mmap_mode="r")
    idx = np.where(DOWN)[0]
    k = int(np.where((tg[idx[0]] == (9, 4250)).all(1))[0][0])
    assert (tg[idx, k] == (9, 4250)).all(), "#4250 not in the same target slot on every down clip"
    return 100 * at[idx, k, :, :-1].astype(np.float32).sum(0) / prof[idx, 0, :, 4250].astype(np.float32).sum()


def real_profiles(L):
    """(12, F) mean activation on down clips: L5 from the PE probe's 8 down clips, L7 / L9 from the 311 (baseline arm)."""
    if L == 5:
        return np.load(A / "umt_pe_probe/real_43.npz")["full"].astype(float).mean(0)
    return np.load(A / f"umt_clock_downstream_L5_raw/prof_L{L}_R.npy", mmap_mode="r")[np.where(DOWN)[0], 0].astype(float).mean(0)


def table(L, n=20):
    """Top-n inputs of #4250 at layer L by |total linear attribution|, with blank, real, best synth and bucket."""
    at, real = attr_by_frame(L), real_profiles(L)
    blank = np.load(PC / f"L{L}.npz")["blank_full"][0].astype(float).sum(0)
    synth = np.load(PC / f"synth_L{L}.npz")["synth"].astype(float).sum(1).max(0)
    f = np.argsort(-np.abs(at.sum(0)))[:n]
    d = pd.DataFrame({"feature": f, "attr": at.sum(0)[f], "blank": blank[f], "real": real.sum(0)[f], "synth": synth[f]})
    # 20/11: bucket and blank / real from the dictionary-wide characterisation (cut 0.8, real over 696 clips), so the
    # #4250 inputs use the same rule as the layer-wide DFA split; down-clip real level kept in 'real' for the profiles
    feat = pd.read_csv(PC / f"features_L{L}.csv").set_index("latent")
    d["bucket"] = feat.cls.reindex(f).map({"pe": "pure PE", "mixed": "mixed", "content": "content"}).values
    d["blank_real_dict"] = feat.blank_real.reindex(f).values
    return d.assign(layer=L), at, real


def panel(ax, d, curve):
    """One line per input: builds blue, hold-downs orange; end label = #id and total linear attribution %."""
    for _, r in d.iterrows():
        f, c = int(r.feature), POS if r.attr > 0 else NEG
        y = curve(f); ax.plot(range(12), y, color=c, lw=1.6)
        ax.annotate(f"#{f} ({r.attr:+.0f}%)", (11, y[-1]), fontsize=7, color=c, xytext=(3, 0), textcoords="offset points", va="center")
    ax.set_xlim(0, 13.5); ax.spines[["top", "right"]].set_visible(False)


def layer_figure(L):
    d, at, real = table(L)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for i, b in enumerate(["pure PE", "mixed", "content"]):
        rows = d[d.bucket == b]
        panel(axes[i, 0], rows, lambda f: real[:, f] / max(real[:, f].max(), 1e-9))
        panel(axes[i, 1], rows, lambda f: at[:, f]); axes[i, 1].axhline(0, color="grey", lw=0.5)
        axes[i, 0].set_ylabel(f"{b} (n={len(rows)})\nactivation, max = 1"); axes[i, 1].set_ylabel("% of #4250 per frame")
    axes[0, 0].set_title("Activation on down clips", fontsize=10); axes[0, 1].set_title("When it feeds #4250 (linear attribution per frame)", fontsize=10)
    for ax in axes[2]: ax.set_xlabel("frame")
    fig.suptitle(f"#4250 (down): top-20 L{L} inputs by bucket", fontsize=11); fig.tight_layout(); fig.savefig(OUT / f"inputs_4250_L{L}.png", dpi=130)
    return d


def spectrum(tabs):
    """One row per layer: inputs at log10(blank / real) vs linear attribution; L9 row = #4250 itself."""
    x9 = np.log10(max(pd.read_csv(PC / "features_L9.csv").set_index("latent").blank_real[4250], 1e-3))   # dictionary rule
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, d in zip(axes[:2], tabs):
        x = np.log10(np.maximum(d.blank_real_dict, 1e-3))                  # dictionary blank / real, floor = -3
        for xi, yi, f in zip(x, d.attr, d.feature):
            c = POS if yi > 0 else NEG; ax.scatter(xi, yi, s=40, color=c, zorder=3)
            ax.annotate(f"#{f}", (xi, yi), fontsize=7, color=c, xytext=(4, 3), textcoords="offset points")
        ax.set_ylabel(f"L{d.layer.iloc[0]}: % of #4250\n(linear attribution)")
    axes[2].scatter(x9, 0, s=80, color="k"); axes[2].annotate("#4250 itself", (x9, 0), xytext=(6, 6), textcoords="offset points", fontsize=8)
    axes[2].set_yticks([]); axes[2].set_ylabel("L9")
    for ax in axes:
        for v in (np.log10(0.01), np.log10(0.8)): ax.axvline(v, color="grey", lw=0.8, ls=":")
        ax.axhline(0, color="grey", lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("PE spectrum of #4250's inputs (dotted: blank = 1% real | blank = 0.8 x real)", fontsize=10)
    axes[2].invert_xaxis()                                     # shared x: flips all three rows; pure PE on the left
    axes[2].set_xlabel("log10(blank / real): pure PE  <-  ->  content"); fig.tight_layout(); fig.savefig(OUT / "spectrum_4250_layers.png", dpi=130)


if __name__ == "__main__":
    tabs = [layer_figure(L) for L in (5, 7)]
    spectrum(tabs)
    pd.concat(tabs).round(2).to_csv(OUT / "inputs_4250_layers.csv", index=False)
