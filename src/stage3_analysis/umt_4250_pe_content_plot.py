"""PE vs content map of the #4250 sheet features (18/11/26). x = blank / real (what the PE alone gives, log);
y = (best synthetic - blank) / real (what one synthetic object adds on top of the PE); size = % of #4250 removed
by single ablation (R, spliced group run). Out: outputs/analysis/umt_4250_feature_checks/pe_content_map.png
Usage: uv run python src/stage3_analysis/umt_4250_pe_content_plot.py"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
A = Path("outputs/analysis")
CFG = {
    "sheet": A / "feature_sheets/UMT_4250_feature_analysis.csv",
    "abl": A / "umt_group_ablation_L5_ud_splice/analysis/group_ablation.csv",
    "out": A / "umt_4250_feature_checks/pe_content_map.png",
    "colors": {"PE driven": "#2a78d6", "Mixed (PE + content)": "#eb6834", "Content only": "#1baf7a"},
    "x_floor": 1e-3, "label_min": 2.0,          # zero-blank features drawn at the floor; label if ablation >= 2%
}


def table(cfg):
    s = pd.read_csv(cfg["sheet"], encoding="utf-8-sig")
    s["bucket"] = s[list(cfg["colors"])].eq("x").idxmax(axis=1)
    r = s["Real act (8 down clips)"]
    s["x"] = np.maximum(s["Blank act (PE alone)"] / r, cfg["x_floor"])
    s["y"] = (s["Best synthetic act"] - s["Blank act (PE alone)"]) / r
    s["abl"] = s["% of 4250 removed (ablation, R)"]           # sheet holds the latest run's single ablations
    return s


def main(cfg):
    s = table(cfg)
    fig, ax = plt.subplots(figsize=(9, 6))
    for b, c in cfg["colors"].items():
        g = s[s.bucket == b]
        ax.scatter(g.x, g.y, s=30 + 12 * g.abl.clip(lower=0), c=c, edgecolors="white", linewidths=1.5, label=b, zorder=3)
    for _, r in s[s.abl.abs() >= cfg["label_min"]].iterrows():
        ax.annotate(f"#{r['Feature Idx']} ({r.abl:.0f}%)", (r.x, r.y), xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="#333333")
    ax.axvline(1, color="#999999", lw=0.8, ls="--"); ax.axvline(0.01, color="#999999", lw=0.8, ls="--")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("PE alone: blank / real activation (log; 1 = PE gives the full real level)")
    ax.set_ylabel("Synthetic object adds: (best synthetic − blank) / real")
    ax.set_title("#4250 features: PE vs content (size = % of #4250 removed by single ablation)", fontsize=10)
    lg = ax.legend(frameon=False, fontsize=8, loc="upper left")
    for h in lg.legend_handles:
        h.set_sizes([60])                                   # equal legend markers (point size encodes ablation)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(cfg["out"], dpi=150)
    print(s[["Feature Idx", "bucket", "x", "y", "abl"]].sort_values("x").round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
