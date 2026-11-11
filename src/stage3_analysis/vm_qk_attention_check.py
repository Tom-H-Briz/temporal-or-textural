"""
Do VM block L+1's heads attend by tubelet on real clips, as the QK resonance says (same: 6/9/10/11, fwd t->t+1: 1/7,
back t->t-1: 2)? Hooks layernorm_before, recomputes that block's softmax(QK^T) per head, collapses to tubelet x tubelet
(row = query tubelet, rows sum to 1; uniform = 1/8). Random SSv2 val clips. CPU / MPS.
Usage: SAE_LAYER=5 [N_CLIPS=32] uv run python src/stage3_analysis/vm_qk_attention_check.py
"""
import os
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "notebooks")]
from ToT_utils import (CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY, SSv2ClipDataset,
                       _strip_brackets, get_processor, load_metadata)

L = int(os.environ.get("SAE_LAYER", 5))
CFG = {
    "attn_block": L + 1, "n_clips": int(os.environ.get("N_CLIPS", 32)), "seed": 0,
    "n_heads": 12, "n_tub": 8, "n_spatial": 196,
    "device": "mps" if torch.backends.mps.is_available() else "cpu",
    "out_dir": ROOT / "outputs/analysis/vm_qk_resonance",
}


def pick_clips(cfg):
    """n_clips random SSv2 val clips (seeded): paths and class ids."""
    d = DATASET_REGISTRY["ssv2"]
    label_map, clips, _ = load_metadata(d["labels_path"], d["validation_path"])
    clips = random.Random(cfg["seed"]).sample(clips, cfg["n_clips"])
    paths = [Path(d["video_dir"]) / f"{c['id']}.webm" for c in clips]
    return paths, [label_map[_strip_brackets(c["template"])] for c in clips]


def tubelet_attention(x, a, cfg):
    """x (B, N, 768) = layernorm_before output -> (B, H, 8, 8): sum over key patches, mean over query patches."""
    B, N, _ = x.shape
    H, T, S = cfg["n_heads"], cfg["n_tub"], cfg["n_spatial"]
    q = (x @ a.query.weight.T + a.q_bias).view(B, N, H, -1).transpose(1, 2)
    k = (x @ a.key.weight.T).view(B, N, H, -1).transpose(1, 2)              # no k bias in VideoMAE
    p = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1)  # (B, H, N, N)
    return p.view(B, H, T, S, T, S).sum(-1).mean(3)


def run(cfg, paths):
    """(n_clips, H, 8, 8) tubelet attention of block attn_block, one clip at a time."""
    m, ck = MODEL_REGISTRY["videomae"], CHECKPOINT_REGISTRY[("videomae", "ssv2")]
    model = m["model_class"].from_pretrained(ck).to(cfg["device"]).eval()
    blk, acc = model.videomae.encoder.layer[cfg["attn_block"]], []
    blk.layernorm_before.register_forward_hook(
        lambda mod, i, o: acc.append(tubelet_attention(o, blk.attention.attention, cfg).cpu()))
    ds = SSv2ClipDataset(paths, get_processor(m, ck), m["num_frames"])
    with torch.no_grad():
        for i in range(len(ds)):
            model(pixel_values=ds[i].unsqueeze(0).to(cfg["device"]))
    return torch.cat(acc)


def summary(A):
    """Per head, clip-mean attention to the same tubelet, t+1 and t-1 (diagonal means; uniform = 0.125)."""
    import pandas as pd
    M = A.mean(0)
    d = lambda o: M.diagonal(offset=o, dim1=1, dim2=2).mean(1).numpy()
    return pd.DataFrame({"same": d(0), "fwd_t+1": d(1), "back_t-1": d(-1)}).rename_axis("head").round(3)


def main(cfg):
    paths, labels = pick_clips(cfg)
    A = run(cfg, paths)
    torch.save({"attn": A, "clips": [p.stem for p in paths], "labels": labels},
               cfg["out_dir"] / f"tubelet_attention_block{cfg['attn_block']}.pt")
    print(summary(A).to_string())
    for h in (1, 7, 2, 6):
        print(f"\nhead {h} (row = query tubelet, col = key tubelet)\n{A.mean(0)[h].numpy().round(2)}")


if __name__ == "__main__":
    main(CFG)
