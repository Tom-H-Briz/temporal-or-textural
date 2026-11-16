"""
Synthetic-motion builders + clock (16/11/26): the builder-finding method on the motion_vid_generator clips.
Raw stream at L5. Per clip: (a) attribution (activation x gradient) of the L7/L9 down/up/camera targets to every
L5 latent per frame; (b) L5 2x2 arms (clock / builders incl. #5519 / both vs batch_base) -> VTC logits and
L7/L9 profiles of all latents. #4250 is never edited, only measured. Reuses umt_clock_downstream_ablation.
Usage: uv run python -u src/stage3_analysis/umt_synthetic_motion_attr.py   (run the generator first)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
os.environ.setdefault("LAYER", "5")
import umt_clock_downstream_ablation as dsa

SETS = json.load(open(ROOT / "src/stage3_analysis/umt_clock_downstream_sets.json"))["5"]
CFG = dict(dsa.CFG, layer=5, capture=[7, 9], splice=False, attr=True, attr_batch=4, edit_batch=4,
           clip_dir=ROOT / "outputs/synthetic_motion/directional_clips",
           out_dir=Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/umt_synthetic_motion_attr"))),
           builders=[5135, 4638, 186, 5519],      # 13/11 builders + the up-side primitive #5519 (user, 16/11)
           targets=[(9, 4250), (9, 1375), (9, 393), (9, 4144), (9, 1684), (9, 3647), (9, 2181), (9, 2472),
                    (9, 3884), (7, 1029), (7, 344)])


def arms(cfg):
    """[(arm, feats)]: baseline, batch_base (empty edit in a batched pass), clock, builders, both."""
    clock = SETS["clock"]
    return [("baseline", []), ("batch_base", []), ("clock", clock),
            ("builders", cfg["builders"]), ("both", clock + cfg["builders"])]


def one(st, cfg, pv, arm_list):
    """One clip -> logits (R, 174), profiles {7, 9: (R, 12, F)}, attribution (K, 12, F+1) for cfg targets."""
    import umt_vtc_clock_ablation as clock
    lg, prof, *_ = dsa.one_clip(st, dict(cfg, attr=False), pv, arm_list, SETS["comb"], cls=-1)
    with torch.no_grad():
        _, x, z0 = clock.clip_pass(st, pv)
    attr = dsa.attribute(st, cfg, pv, x, z0, cfg["targets"])
    return lg.cpu(), {c: p.cpu() for c, p in prof.items()}, attr.half().cpu()


def main(cfg):
    meta = json.load(open(cfg["clip_dir"] / "metadata.json"))[: cfg["max_clips"] or None]   # MAX_CLIPS=2: smoke
    batch = torch.load(cfg["clip_dir"] / "all_umt_clips.pt")              # (N, T, C, H, W)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    arm_list, st = arms(cfg), dsa.setup(cfg)
    lg, prof, attr = [], {7: [], 9: []}, []
    for i, m in enumerate(meta):
        a, p, t = one(st, cfg, batch[i:i + 1].to(cfg["device"]), arm_list)
        lg.append(a); attr.append(t)
        for c in prof:
            prof[c].append(p[c].half())
        print(f"  {i + 1}/{len(meta)} {m['name']}", flush=True)
    np.save(cfg["out_dir"] / "logits.npy", torch.stack(lg).float().numpy())        # (N, R, 174)
    np.save(cfg["out_dir"] / "attr.npy", torch.stack(attr).numpy())                 # (N, K, 12, F+1)
    for c in prof:
        np.save(cfg["out_dir"] / f"prof_L{c}.npy", torch.stack(prof[c]).numpy())   # (N, R, 12, F)
    json.dump({"arms": arm_list, "targets": cfg["targets"], "clips": [m["name"] for m in meta]},
              open(cfg["out_dir"] / "index.json", "w"))


if __name__ == "__main__":
    main(CFG)
