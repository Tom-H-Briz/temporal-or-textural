"""
Plot feature firing frequency distributions for the k=64 and k=128 SAE runs.

Usage: uv run python notebooks/plot_feature_counts.py
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
COUNTS_DIR = ROOT / "outputs"

VAL_TOKENS = 4_000 * 1_568  # val clips × tokens per clip

files = {
    "k=64":  COUNTS_DIR / "feature_counts_job64.pt",
    "k=128": COUNTS_DIR / "feature_counts_job128.pt",
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, (label, path) in zip(axes, files.items()):
    counts = torch.load(path, weights_only=True).float()
    firing_rate = counts / VAL_TOKENS

    ax.hist(firing_rate.numpy(), bins=50, log=True)
    ax.set_xlabel("Fraction of val tokens")
    ax.set_title(label)
    ax.set_xlim(left=0)

    dead = int((counts == 0).sum())
    ax.text(0.97, 0.95, f"Dead: {dead}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)

axes[0].set_ylabel("Number of features (log scale)")

fig.suptitle("Feature firing frequency — val set")
fig.tight_layout()

out = ROOT / "outputs" / "feature_freq_histogram.png"
fig.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
