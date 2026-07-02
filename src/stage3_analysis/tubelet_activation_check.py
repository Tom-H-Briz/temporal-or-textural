"""
Per-tubelet activation check — feature 990, 3 specific clips.
Forward pass only. One-off sanity check, not pipeline infrastructure.
"""

import sys
from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.dfa_engine import DFAEngine

FEATURE_IDX = 990
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

SAE_PATH  = str(ROOT / "outputs/sae/sae_layer7_job64.pt")
DMEAN     = str(ROOT / "outputs/sae/layer7_dim_mean.pt")

CLIPS = [
    ("123895", 32,  ROOT / "data/ssv2/20bn-something-something-v2/123895.webm"),
    ("153893", 143, ROOT / "data/ssv2/20bn-something-something-v2/153893.webm"),
    ("62283",  40,  ROOT / "data/ssv2/20bn-something-something-v2/62283.webm"),
]

with DFAEngine("videomae", SAE_PATH, DMEAN, layer=7, device=DEVICE, sae_k=64) as engine:
    for clip_id, class_id, clip_path in CLIPS:
        z    = engine.get_z(clip_path)           # (1568, dict_size)
        feat = z[:, FEATURE_IDX]                 # (1568,)
        per_tubelet = feat.reshape(8, 196).sum(dim=1)   # (8,)
        share       = per_tubelet.abs() / per_tubelet.abs().sum()

        print(f"\nclip{clip_id} (class {class_id}):")
        for t in range(8):
            print(f"  tubelet {t}: sum={per_tubelet[t].item():+.4f}  share={share[t].item():.4f}")
