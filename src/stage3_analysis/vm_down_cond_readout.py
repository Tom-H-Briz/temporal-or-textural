"""VM real-clip condition readout (19/11/26): the same top-20 L9 down candidates as vm_synthetic_down_readout.py,
on real down (43) / up (45) clips under R, C1 (tubelet pairs shuffled) and Rev (R flipped in time).
Out: outputs/analysis/vm_down_cond_readout/profiles.npz (cond -> (clips, 8, 20)) + clips.csv.
Usage: [MAX_CLIPS=5] uv run python src/stage3_analysis/vm_down_cond_readout.py"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from vm_synthetic_down_readout import CFG as SCFG, top_features
CFG = dict(SCFG, conds=("R", "C1", "Rev"), num_workers=0,
           classes=tuple(int(c) for c in os.environ.get("CLASSES", "43,45").split(",")),   # 168,165 = camera up / down
           max_clips=int(os.environ.get("MAX_CLIPS", 0)),                 # 0 = all clips of each class
           clips_csv=SCFG["real"] / "clips.csv",                          # same 311 down / 359 up clips as the 11/11 run
           video_dir=os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
           out_dir=Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/vm_down_cond_readout"))))


def clip_table(cfg):
    """Down / up clips from the 11/11 run (optionally the first max_clips per class)."""
    c = pd.read_csv(cfg["clips_csv"])
    c = c[c.class_id.isin(cfg["classes"])]
    if cfg["max_clips"]:
        c = c.groupby("class_id").head(cfg["max_clips"])
    return c.reset_index(drop=True)


def codes(st, pv, feats, cfg):
    """One forward -> L9 SAE codes of feats summed over space per tubelet: (8, n_feats)."""
    with torch.no_grad():
        st["model"](pixel_values=pv.to(cfg["device"]))
        z = st["sae"].encode(st["x"][0].float() - st["dm"])[1][:, feats]
    return z.reshape(8, -1, len(feats)).sum(1).cpu().numpy()


def main(cfg):
    import vision_pair_phase3
    from vm_object_contrast import clip_loader
    feats = top_features(cfg).latent.astype(int).tolist()
    clips, st = clip_table(cfg), vision_pair_phase3.setup(cfg)
    out = {c: [] for c in cfg["conds"]}
    loaders = zip(clip_loader(clips, cfg, "R", st["processor"]), clip_loader(clips, cfg, "C1", st["processor"]))
    for n, ((pv, _), (pv_c1, _)) in enumerate(loaders):
        out["R"].append(codes(st, pv, feats, cfg))
        out["C1"].append(codes(st, pv_c1, feats, cfg))
        out["Rev"].append(codes(st, pv.flip(1), feats, cfg))                # (1, T, C, H, W): flip time
        if n % 50 == 0:
            print(f"{n}/{len(clips)}", flush=True)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    np.savez(cfg["out_dir"] / "profiles.npz", latents=np.array(feats), **{c: np.stack(v) for c, v in out.items()})
    clips.to_csv(cfg["out_dir"] / "clips.csv", index=False)


if __name__ == "__main__":
    main(CFG)
