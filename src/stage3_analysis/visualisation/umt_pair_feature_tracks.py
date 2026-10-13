"""Per-feature motion figures for the UMT up/down vision pair contrast (Stage 1).

For each feature: its 3 most-active "move up" clips and 3 most-active "move down" clips.
Per clip, two rows of the 12 frames UMT saw, with the feature's 14x14 activation overlaid and
its activation-weighted centroid marked:
  R    real order
  Rev  the same frames reversed (what the model saw under time reversal)
plus a final panel: centroid row vs frame for R and Rev (up = towards top of image).
Reads the stored sparse codes (umt_vision_pair_contrast.py), so no model is needed; frames
are decoded locally with UMT's own 'middle' sampling and squashed to 224 like UMTProcessor.
Usage: uv run --with easydict --with "timm==0.4.12" python src/stage3_analysis/visualisation/umt_pair_feature_tracks.py
"""
import os
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "notebooks"), str(Path(__file__).parent)]
from feature_vis_vm import _overlay   # shared signed-overlay blend (RdBu_r, alpha 0.45)

CFG = {
    # UMT only. PAIR="a,b" as the extractor (default 45,43 = up/down; others get a _a_b dir)
    "pair":      tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(",")),
    "video_dir": ROOT / "data/ssv2_val_set",
    "features":  [4250, 393, 6127, 3434, 2782, 2642, 5473, 815, 5901],
    "n_clips":   3,                             # per class
    "n_frames":  12, "grid": 14, "img": 224,
}
CFG["in_dir"] = ROOT / "outputs/analysis" / ("umt_vision_pair_contrast" + (
    "" if CFG["pair"] == (45, 43) else f"_{CFG['pair'][0]}_{CFG['pair'][1]}"))
# centroid axis for the side track panel: vertical (row) only makes sense for up/down;
# closer/apart-type pairs move horizontally in the image plane
CFG["track_axis"] = "row" if CFG["pair"] == (45, 43) else "col"


def load_frames(clip_id, cfg):
    """The 12 frames UMT saw: 'middle' sampling, squashed to 224x224 (uint8, for display)."""
    from umt_wrapper import sample_frames_umt
    with av.open(str(cfg["video_dir"] / f"{clip_id}.webm")) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    idx = sample_frames_umt(len(frames), cfg["n_frames"])
    return [np.asarray(Image.fromarray(frames[i]).resize((cfg["img"],) * 2, Image.BICUBIC)) for i in idx]


def feature_maps(clip_id, cond, f, cls, cfg):
    """(frames, 14, 14) activation of latent f from the stored sparse codes, plus the sign
    convention of feature_vis_vm: activation x sign(this clip's DFA for f, THIS condition).
    DFA here is toward up; flipped for down clips, so + (red) = pushes toward the true class."""
    s = np.load(cfg["in_dir"] / "sparse" / cond / f"{clip_id}.npz")
    m = s["lat"] == f
    out = np.zeros(cfg["n_frames"] * cfg["grid"] ** 2, np.float32)
    out[s["tok"][m].astype(np.int64)] = s["z"][m].astype(np.float32)
    toward_true = float(s["dfa"][m].sum()) * (1.0 if cls == cfg["pair"][0] else -1.0)
    return out.reshape(cfg["n_frames"], cfg["grid"], cfg["grid"]), (np.sign(toward_true) or 1.0)


def centroids(maps):
    """Activation-weighted (row, col) per frame in patch units; NaN where inactive."""
    g = maps.shape[1]
    tot = maps.sum((1, 2))
    rr, cc = np.mgrid[0:g, 0:g]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(tot > 0, (maps * rr).sum((1, 2)) / tot, np.nan)
        c = np.where(tot > 0, (maps * cc).sum((1, 2)) / tot, np.nan)
    return r, c


def pick_clips(f, cfg):
    """Top-n clips by total activation of f, separately among up and down clips (R)."""
    arr = np.load(cfg["in_dir"] / "clip_feature_arrays_R.npz")
    pc = pd.read_parquet(cfg["in_dir"] / "per_clip.parquet")
    lab = pc[pc.condition == "R"].set_index("clip_id").true_class
    ids, z = arr["clip_id"], arr["zsum"][:, f]
    keep = np.ones(len(ids), bool)
    if cfg.get("only_vertical"):    # clips with clear image-plane vertical motion (optical flow)
        mf = pd.read_parquet(cfg["in_dir"] / "motion_flow.parquet").set_index("clip_id")
        keep = mf.vertical.loc[ids].to_numpy()
    out = []
    for cls in cfg["pair"]:
        sel = np.where((lab.loc[ids].to_numpy() == cls) & keep)[0]
        out += [(str(ids[i]), cls) for i in sel[np.argsort(-z[sel])[: cfg["n_clips"]]]]
    return out


def draw_row(axes, frames, maps, sign, norm, label, cfg):
    """12 frame panels: signed overlay (feature_vis_vm._overlay, RdBu_r: red = toward the true
    class, blue = against) + centroid of the unsigned activation."""
    import matplotlib.cm as cm
    up = cfg["img"] // cfg["grid"]
    r, c = centroids(maps)
    for t, ax in enumerate(axes):
        ax.imshow(_overlay(frames[t], maps[t] * sign, norm, cm.RdBu_r))
        if not np.isnan(r[t]):
            ax.plot(c[t] * up + up / 2, r[t] * up + up / 2, "o", ms=5, mfc="cyan", mec="black")
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    return c if cfg["track_axis"] == "col" else r


def draw_track(ax, r_real, r_rev, cfg):
    """Centroid (row for up/down, col otherwise) vs frame; R solid, Rev dashed. Rows are
    inverted so 'up' is up; columns ascend left->right of the image."""
    t = np.arange(cfg["n_frames"])
    ax.plot(t, r_real, "-o", ms=3, color="tab:blue", label="R")
    ax.plot(t, r_rev, "--o", ms=3, color="tab:orange", label="Rev")
    ax.set_ylim(cfg["grid"] - 0.5, -0.5) if cfg["track_axis"] == "row" else ax.set_ylim(-0.5, cfg["grid"] - 0.5)
    ax.set_ylabel(f"centroid {cfg['track_axis']}", fontsize=6)
    ax.set_xticks([0, 11]); ax.tick_params(labelsize=6)


def make_figure(f, ft, cfg):
    """One PNG per feature: 2 rows (R, Rev) per clip, 3 up-clips then 3 down-clips."""
    clips = pick_clips(f, cfg)
    import matplotlib.colors as mcolors
    data = [(cid, cls, load_frames(cid, cfg), feature_maps(cid, "R", f, cls, cfg),
             feature_maps(cid, "Rev", f, cls, cfg)) for cid, cls in clips]
    # feature_vis_vm convention: symmetric norm at the 99th pct of |R maps| (here: shown clips)
    vmax = float(np.percentile(np.abs(np.stack([m for *_, (m, _), _ in data])), 99)) or 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    fig, axes = plt.subplots(2 * len(data), cfg["n_frames"] + 1, figsize=(17, 1.45 * 2 * len(data)),
                             gridspec_kw={"width_ratios": [1] * cfg["n_frames"] + [1.6]})
    for i, (cid, cls, frames, (m, s), (mv, sv)) in enumerate(data):
        name = "A" if cls == cfg["pair"][0] else "B"
        r_real = draw_row(axes[2 * i, :-1], frames, m, s, norm, f"{name} {cid}\nR  sign {int(s):+d}", cfg)
        r_rev = draw_row(axes[2 * i + 1, :-1], frames[::-1], mv, sv, norm, f"Rev  sign {int(sv):+d}", cfg)
        draw_track(axes[2 * i, -1], r_real, r_rev, cfg)
        axes[2 * i + 1, -1].axis("off")
    axes[0, -1].legend(fontsize=6, loc="upper right")
    row = ft.set_index("feature").loc[f]
    # optional columns (track/slope) exist only when the vertical motion-flow step ran
    extra = "".join(f"  track {row[k]:.2f}" for k in ("track_score",) if k in row.index)
    extra += "".join(f"  slope {k} {row[k]:+.2f}" for k in ("slope_up", "slope_down") if k in row.index)
    fig.suptitle(f"UMT vision L9 feature {f}  |  evidence {row.evidence:.3f}  bias {row.bias:.3f}  "
                 f"AUROC(A>B) {row.auroc:.2f}{extra}", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.01, wspace=0.03, hspace=0.06)
    # dedicated colorbar axis (feature_vis_vm: never ax=axes, it fights subplots_adjust)
    cax = fig.add_axes([0.965, 0.35, 0.008, 0.3])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm), cax=cax)
    cbar.set_ticks([-vmax, 0, vmax]); cbar.set_ticklabels([f"{-vmax:.1f}", "0", f"{vmax:.1f}"])
    cbar.set_label("signed activation (+ toward true class)", fontsize=8)
    return fig


def main():
    """Optional args: feature ids to render, and --vertical to draw only clips with clear
    image-plane vertical motion (umt_pair_motion_flow.py); those PNGs get a _vertical suffix."""
    cfg = dict(CFG)
    args = sys.argv[1:]
    cfg["only_vertical"] = "--vertical" in args
    cfg["features"] = [int(a) for a in args if a.isdigit()] or cfg["features"]
    ft = pd.read_csv(cfg["in_dir"] / "feature_table.csv")
    out = cfg["in_dir"] / "figures"
    out.mkdir(exist_ok=True)
    for f in cfg["features"]:
        fig = make_figure(f, ft, cfg)
        name = f"feature_{f}{'_vertical' if cfg['only_vertical'] else ''}.png"
        fig.savefig(out / name, dpi=110)
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
