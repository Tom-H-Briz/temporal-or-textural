"""Slice the target latents out of the top-20 group-ablation prof files (19/11/26), for local clip-bootstrap CIs.
prof_L{c}_{cond}.npy (n, rows, T, F) -> extract/prof_L{c}_{cond}.npy (n, rows, T, k), targets in CFG order.
Usage: uv run python src/stage3_analysis/umt_top20_extract_targets.py   (Isambard; reads in clip chunks)"""
from pathlib import Path
import numpy as np

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
CFG = {
    "runs": {"umt_top20_4250_L5_ud_splice": {9: [4250, 393]}, "umt_top20_push_5083_L5_splice": {9: [5083]},
             "umt_top20_behind_491_L5_splice": {9: [491, 768]}, "umt_top20_front_768_L5_splice": {9: [491, 768]},
             "umt_top20_ball_713_L5_splice": {9: [713], 7: [4973]},
             "umt_top20_paper_4228_L5_splice": {9: [4228], 7: [5960]},
             "umt_top20_cup_6011_L5_splice": {9: [6011], 7: [2392]}},
    "conds": ("R", "Rev"),
    "chunk": 32,
}


def extract(src, dst, latents, chunk):
    """One pass over src in clip chunks; float32 out, (n, rows, T, k)."""
    prof = np.load(src, mmap_mode="r")
    out = np.zeros(prof.shape[:3] + (len(latents),), dtype=np.float32)
    for i in range(0, prof.shape[0], chunk):
        out[i:i + chunk] = np.asarray(prof[i:i + chunk])[..., latents]
    np.save(dst, out)
    print(f"{dst} {out.shape}", flush=True)


def main(cfg):
    for run, targets in cfg["runs"].items():
        (A / run / "extract").mkdir(exist_ok=True)
        for c, lat in targets.items():
            for cond in cfg["conds"]:
                extract(A / run / f"prof_L{c}_{cond}.npy", A / run / "extract" / f"prof_L{c}_{cond}.npy", lat, cfg["chunk"])


if __name__ == "__main__":
    main(CFG)
