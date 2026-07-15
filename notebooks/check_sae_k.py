"""
Read-only check: does a VM SAE checkpoint have sae_k saved?

Verifies whether patch_checkpoints_add_sae_k.py-style backfilling is actually
needed for the layer 5/9 VM checkpoints trained via train_sae_vm_l5_l9.sh —
does not modify anything.

Usage:
    uv run python notebooks/check_sae_k.py
"""

from pathlib import Path
import torch

ROOT    = Path(__file__).parent.parent
SAE_DIR = ROOT / "outputs" / "sae"
LAYERS  = [5, 9]
JOB     = "64"

for layer in LAYERS:
    path = SAE_DIR / f"sae_layer{layer}_job{JOB}.pt"
    if not path.exists():
        print(f"  {path.name}: NOT FOUND")
        continue
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    is_wrapped = isinstance(ckpt, dict) and "sae_state_dict" in ckpt
    sae_k = ckpt.get("sae_k") if isinstance(ckpt, dict) else None
    print(f"  {path.name}: wrapped={is_wrapped}  sae_k={sae_k!r}")
