"""
One-time patch: add sae_k=64 to all TF _best.pt checkpoints that lack it.
Run once on Isambard, then delete.

Usage: uv run python notebooks/patch_checkpoints_add_sae_k.py
"""

import sys
from pathlib import Path

import torch

ROOT    = Path(__file__).parent.parent
SAE_DIR = ROOT / "outputs" / "sae"
SAE_K   = 64

for ckpt_path in sorted(SAE_DIR.glob("sae_tf_*_best.pt")):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "sae_k" in ckpt:
        print(f"  SKIP {ckpt_path.name}  (already has sae_k={ckpt['sae_k']})")
        continue
    ckpt["sae_k"] = SAE_K
    torch.save(ckpt, ckpt_path)
    print(f"  PATCHED {ckpt_path.name}  sae_k={SAE_K}")

print("Done.")
