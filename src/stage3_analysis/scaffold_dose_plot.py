"""
Plot the scaffold dose-response: model accuracy vs latent scale alpha.

X axis: alpha (categorical, equal spacing — x0 x0.5 x1 x2 x4).
Y axis: top-1 accuracy (%). Splice-space curve (5 alphas) with the raw-stream
points (x0/x1/x2) overlaid — the inverted-U with a broad flat optimum.

Usage: uv run python src/stage3_analysis/scaffold_dose_plot.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CFG = {
    "splice_csv": ROOT / "outputs/analysis/scaffold_dose_response/scaffold_dose_summary.csv",
    "raw_csv": ROOT / "outputs/analysis/scaffold_dose_raw/scaffold_dose_raw_summary.csv",
    "out_png": ROOT / "outputs/analysis/scaffold_dose_response/scaffold_dose_curve.png",
    "alphas": [0.0, 0.5, 1.0, 2.0, 4.0],
}


def acc_at(csv: Path, alphas: list[float]) -> dict[float, float]:
    overall = pd.read_csv(csv).query("scope=='overall'")
    accs = overall.set_index("alpha")["acc_alpha"]
    out = {al: float(accs.loc[al]) * 100 for al in alphas
           if al != 1.0 and al in accs.index}
    if 1.0 in alphas:
        out[1.0] = float(overall.acc_base.iloc[0]) * 100   # base = the x1 arm
    return out


def main() -> None:
    splice = acc_at(CFG["splice_csv"], CFG["alphas"])
    raw = acc_at(CFG["raw_csv"], [0.0, 1.0, 2.0])
    raw_order = [al for al in CFG["alphas"] if al in raw]   # axis order, not dict order
    pos = {al: i for i, al in enumerate(CFG["alphas"])}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    xs = [pos[al] for al in CFG["alphas"]]
    ax.plot(xs, [splice[al] for al in CFG["alphas"]], "o-", color="steelblue",
            lw=2, label="SAE splice (z scaled, decoded)")
    xr = [pos[al] for al in raw_order]
    ax.plot(xr, [raw[al] for al in raw_order], "s--", color="darkorange", lw=1.8,
            label="raw stream (h − (1−α)·Σzᵢdᵢ)")
    for al, acc in splice.items():
        ax.annotate(f"{acc:.1f}", (pos[al], acc), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="steelblue")
    for al, acc in raw.items():
        ax.annotate(f"{acc:.1f}", (pos[al], acc), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9, color="darkorange")
    ax.set_xticks(xs, [f"×{a:g}" for a in CFG["alphas"]])
    ax.set_xlabel("scaffold latent scale α (×1 = model as-is)")
    ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("L5 scaffold dose–response, n=500 representative SL clips")
    ax.set_ylim(60, 88)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", fontsize=9)
    fig.tight_layout()
    fig.savefig(CFG["out_png"], dpi=150)
    print(f"-> {CFG['out_png']}")


if __name__ == "__main__":
    main()
