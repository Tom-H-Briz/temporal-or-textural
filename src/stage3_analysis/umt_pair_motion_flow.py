"""Model-independent image-plane motion for the up/down pair clips (optical flow, OpenCV).

Some "move up" clips are filmed from above: the object rises TOWARDS the camera, so it grows
rather than moving up the frame, and vertical centroid tracking cannot apply. Per clip, on the
same 12 frames UMT saw (middle sampling, 224 squash, greyscale), Farneback flow between
consecutive frames, minus each pair's median flow (camera motion):
  vy_cum     summed mean vertical flow of the moving pixels (px; negative = up the frame)
  expansion  summed mean radial flow about the moving region's centre (+ = growing)
  moving     mean fraction of pixels above the motion threshold
Usage: uv run --with opencv-python-headless --with easydict --with "timm==0.4.12" \
         python src/stage3_analysis/umt_pair_motion_flow.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "notebooks"), str(ROOT / "src" / "stage3_analysis" / "visualisation")]
CFG = {
    "in_dir":     ROOT / "outputs/analysis/umt_vision_pair_contrast",
    "move_px":    1.0,    # |flow| (px/frame) above which a pixel counts as moving
}


def clip_flow(frames, move_px):
    """frames: 12 RGB uint8 (224x224) -> (vy_cum, expansion, moving_frac) for the clip."""
    import cv2
    grey = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    vy_cum, exp_cum, moving = 0.0, 0.0, []
    for a, b in zip(grey[:-1], grey[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fl -= np.median(fl.reshape(-1, 2), 0)                  # remove camera motion
        mag = np.linalg.norm(fl, axis=-1)
        m = mag > move_px
        moving.append(m.mean())
        if m.sum() < 50:
            continue
        vy_cum += float(fl[..., 1][m].mean())
        cy, cx = yy[m].mean(), xx[m].mean()
        r = np.stack([xx - cx, yy - cy], -1)
        rn = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-6)
        exp_cum += float((fl * rn).sum(-1)[m].mean())
    return vy_cum, exp_cum, float(np.mean(moving))


def main():
    from umt_pair_feature_tracks import CFG as VCFG, load_frames
    pc = pd.read_parquet(CFG["in_dir"] / "per_clip.parquet")
    r = pc[pc.condition == "R"].reset_index(drop=True)
    rows = []
    for i, (cid, cls) in enumerate(zip(r.clip_id, r.true_class)):
        vy, ex, mv = clip_flow(load_frames(cid, VCFG), CFG["move_px"])
        rows.append(dict(clip_id=cid, true_class=int(cls), vy_cum=vy, expansion=ex, moving=mv))
        if i % 100 == 0:
            print(f"  {i}/{len(r)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(CFG["in_dir"] / "motion_flow.parquet", index=False)
    print(df.groupby("true_class")[["vy_cum", "expansion", "moving"]].describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
