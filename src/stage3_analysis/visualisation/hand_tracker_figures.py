"""VM / TF hand-aligned features (from hand_tracker_alignment.py) vs UMT L9 hand tracker #2782 on the
same up/down clips, drawn on the frames each model saw (8 positions, shortest-edge 224 + centre crop).
Per clip: #2782 mapped to the common grid, then each layer's feature; right panel = centroid row.
Runs the backbone + SAE on CPU (a few clips, seconds). One PNG per model.
Usage: uv run python src/stage3_analysis/visualisation/hand_tracker_figures.py
"""
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("notebooks", "src", "src/stage3_analysis")]
CFG = {"clips": [("86313", "up"), ("72811", "up"), ("110058", "down"), ("142895", "down")],  # #2782's clean trackers
       "features": {"videomae": {5: 4160, 7: 4069, 9: 2347}, "timesformer": {5: 1742, 7: 5573, 9: 4567}},
       "video_dir": ROOT / "data/ssv2_val_set", "out_dir": ROOT / "outputs/analysis/hand_tracker_alignment/figures"}


def frames_seen(cid, model, cfg):
    """8 display frames (VM: first frame of each 2-frame tubelet of 16; TF: its 8), shortest-edge 224 + centre crop.
    Also returns the original width, height (for mapping UMT's full-width columns)."""
    from ToT_utils import sample_frames_ssv2
    with av.open(str(cfg["video_dir"] / f"{cid}.webm")) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = sample_frames_ssv2(len(fr), 16)[::2] if model == "videomae" else sample_frames_ssv2(len(fr), 8)
    out = []
    for i in idx:
        im = Image.fromarray(fr[i]); s = 224 / min(im.size)
        im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        x, y = (im.size[0] - 224) // 2, (im.size[1] - 224) // 2
        out.append(np.asarray(im.crop((x, y, x + 224, y + 224))))
    return out, fr[0].shape[1], fr[0].shape[0]


def feature_maps(model, layer, feat, cfg):
    """{clip_id: (8, 14, 14)} activation of one SAE feature (backbone + SAE at the trained threshold, CPU)."""
    from dfa_engine import DFAEngine
    from ToT_utils import gather_by_position, resolve_sae_checkpoint
    r = resolve_sae_checkpoint(model, layer, dataset_name="ssv2", sae_k=64, job_label="7ep")
    out = {}
    with DFAEngine(model, r["sae_path"], r["dim_mean_path"], layer=layer, device="cpu") as eng:
        for cid, _ in cfg["clips"]:
            z = eng.get_z(cfg["video_dir"] / f"{cid}.webm")[:, feat]
            out[cid] = gather_by_position(z, model).reshape(8, 14, 14).float().numpy()
    return out


def template_map(cid, W, H):
    """UMT L9 #2782 on the common grid (8 bins x crop 14x14), same mapping as the alignment scores."""
    import hand_tracker_alignment as A
    f, tdir = A.CFG["template"]
    return A.load_common(tdir / f"{cid}.npz", "umt", W, H, A.CFG)[0][:, f].reshape(8, 14, 14)


def draw_row(axes, frames, m, vmax, label, colour):
    """8 frames with the map overlaid (Reds, alpha = activation) + centroid; centroid row into axes[8]."""
    tot = m.sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (m * np.arange(14)[None, :, None]).sum((1, 2)) / tot
        c = (m * np.arange(14)[None, None, :]).sum((1, 2)) / tot
    for t in range(8):
        up = np.clip(np.kron(m[t] / vmax, np.ones((16, 16))), 0, 1)
        axes[t].imshow(frames[t]); axes[t].imshow(up, cmap="Reds", vmin=0, vmax=1, alpha=up * .75)
        if not np.isnan(r[t]):
            axes[t].plot(c[t] * 16 + 8, r[t] * 16 + 8, "o", ms=4, mfc="cyan", mec="k")
        axes[t].set_xticks([]); axes[t].set_yticks([])
    axes[0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    axes[8].plot(r, "-o", ms=3, color=colour); axes[8].set_ylim(13.5, -0.5)
    axes[8].set_xticks([0, 7]); axes[8].tick_params(labelsize=6)


def draw(model, rows, frames, cfg):
    """rows: [(label, {clip_id: map})], first = template. Per clip one block of rows; colour scale per row label."""
    nr = len(rows) * len(cfg["clips"])
    fig, ax = plt.subplots(nr, 9, figsize=(15, 1.55 * nr), gridspec_kw={"width_ratios": [1] * 8 + [1.5]})
    vmax = {lab: max(np.percentile(m[c], 99.5) for c, _ in cfg["clips"]) or 1.0 for lab, m in rows}
    for ci, (cid, cls) in enumerate(cfg["clips"]):
        for ri, (lab, maps) in enumerate(rows):
            draw_row(ax[ci * len(rows) + ri], frames[cid], maps[cid], vmax[lab],
                     f"{cid} {cls}\n{lab}" if ri == 0 else lab, "k" if ri == 0 else "tab:red")
    fig.suptitle(f"{model}: best hand-aligned feature per layer vs UMT L9 #2782 (mapped to {model}'s 8 positions + crop);"
                 " right: centroid row (up = top)", fontsize=10)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.96, bottom=0.01, wspace=0.03, hspace=0.08)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out_dir"] / f"hand_{model}.png", dpi=90)
    plt.close(fig)


def main(cfg=CFG):
    for model, feats in cfg["features"].items():
        seen = {cid: frames_seen(cid, model, cfg) for cid, _ in cfg["clips"]}
        rows = [("UMT L9 #2782", {cid: template_map(cid, W, H) for cid, (_, W, H) in seen.items()})]
        rows += [(f"L{L} #{f}", feature_maps(model, L, f, cfg)) for L, f in feats.items()]
        draw(model, rows, {cid: s[0] for cid, s in seen.items()}, cfg)
        print("saved", cfg["out_dir"] / f"hand_{model}.png", flush=True)


if __name__ == "__main__":
    main()
