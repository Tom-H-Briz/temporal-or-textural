"""Frame figure for one L9 SAE feature in the UMT MLM runs: R / C / Rev rows per clip (CPU, local).

Overlay = the feature's activation, signed by its MLM contribution (red = toward the clip's true word).
The C row shows the shuffled frames in the order the model saw them (native-frame shuffle, seed = clip id,
then UMT 'middle' sampling, as PerturbedKineticsDataset), each titled with its native time t (0 = start).
Row labels give the MLM answer and margin T_ab. Clips: CFG["picks"][pair] = (class, condition, quantile
of activation; 1.0 = strongest). Needs umt_mlm_feature_profile.py output and the val videos.
Usage: FEATURE=4250 PAIR=45,43 uv run --with easydict --with "timm==0.4.12" python src/stage3_analysis/visualisation/umt_mlm_feature_frames.py
"""
import os
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "notebooks"), str(Path(__file__).parent)]
from umt_pair_feature_tracks import draw_row, feature_maps   # shared overlay / row drawing
from umt_wrapper import sample_frames_umt

CFG = {
    "feature":   int(os.environ.get("FEATURE", "4250")),
    "pair":      tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(",")),
    "contrast":  ROOT / "outputs/analysis/umt_mlm_feature_contrast",     # sparse codes + per_clip
    "video_dir": ROOT / "data/ssv2_val_set",
    "words":     {(45, 43): ("up", "down"), (44, 41): ("towards", "away")},
    "picks":     {(45, 43): [(43, "R", 1.0), (43, "R", 0.5), (43, "R", 0.25),
                             (45, "Rev", 1.0), (45, "C", 1.0), (45, "C", 0.95)],
                  (44, 41): [(44, "R", 1.0), (44, "R", 0.8), (44, "R", 0.5), (41, "Rev", 1.0)]},
    "n_frames": 12, "grid": 14, "img": 224, "track_axis": "row",
}
CFG["in_dir"] = CFG["contrast"]                                         # feature_maps reads in_dir/sparse
CFG["out_dir"] = ROOT / "outputs/analysis/umt_mlm_feature_profile" / f"f{CFG['feature']}"


def frames_rc(cid, cfg):
    """12 R frames, 12 C frames (as PerturbedKineticsDataset: permute all native frames with
    seed = clip id, then sample), and each C frame's native time in [0, 1]."""
    with av.open(str(cfg["video_dir"] / f"{cid}.webm")) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n, idx = len(fr), sample_frames_umt(len(fr), cfg["n_frames"])
    perm = np.random.default_rng(int(cid)).permutation(n)      # = perturbation.apply_shuffle
    sq = lambda a: np.asarray(Image.fromarray(a).resize((cfg["img"],) * 2, Image.BICUBIC))
    return [sq(fr[i]) for i in idx], [sq(fr[perm[i]]) for i in idx], perm[idx] / (n - 1)


def pick(prof, cls, cond, q):
    """Clip id of class cls at activation quantile q under cond (1.0 = strongest)."""
    d = prof[(prof.true_class == cls) & (prof.condition == cond)].sort_values("z")
    return int(d.clip_id.iloc[min(int(q * len(d)), len(d) - 1)])


def clip_block(axes, cid, cls, margins, norm, cfg):
    """Three rows for one clip: R, C (native time over each frame), Rev (R frames reversed)."""
    fr, fc, t_nat = frames_rc(cid, cfg)
    words = cfg["words"][cfg["pair"]]
    for row, (cond, frames) in enumerate([("R", fr), ("C", fc), ("Rev", fr[::-1])]):
        maps, sign = feature_maps(cid, cond, cfg["feature"], cls, cfg)
        T = margins.loc[(cid, cond)]
        label = f"{cond}: MLM says '{words[0] if T > 0 else words[1]}' ({T:+.1f})"
        if row == 0:
            label = f"{words[cfg['pair'].index(cls)].upper()} clip {cid}\n" + label
        draw_row(axes[row], frames, maps, sign, norm, label, cfg)
        if cond == "C":
            for ax, tn in zip(axes[row], t_nat):
                ax.set_title(f"t={tn:.2f}", fontsize=6, pad=1)


def make_figure(clips, margins, cfg):
    """3 rows per clip, one symmetric colour scale over every map shown."""
    maps = [feature_maps(c, cond, cfg["feature"], k, cfg)[0] for c, k in clips for cond in ("R", "C", "Rev")]
    vmax = float(np.percentile(np.stack(maps), 99.5)) or 1.0
    norm = mcolors.Normalize(-vmax, vmax)
    fig, axes = plt.subplots(3 * len(clips), cfg["n_frames"], figsize=(15, 4.05 * len(clips)))
    for i, (cid, cls) in enumerate(clips):
        clip_block(axes[3 * i:3 * i + 3], cid, cls, margins, norm, cfg)
    fig.subplots_adjust(left=0.15, right=0.95, top=1 - 0.5 / (3 * len(clips)), bottom=0.01, wspace=0.03, hspace=0.18)
    cax = fig.add_axes([0.962, 0.35, 0.008, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm), cax=cax)
    cb.set_label(f"#{cfg['feature']} activation\nred = toward the clip's true word", fontsize=8)
    return fig


def main():
    cfg = CFG
    a, b = cfg["pair"]
    prof = pd.read_parquet(cfg["out_dir"] / "clip_profile.parquet")
    prof = prof[(prof.a == a) & (prof.b == b)]
    margins = prof.set_index(["clip_id", "condition"]).T_ab
    clips = [(pick(prof, cls, cond, q), cls) for cls, cond, q in cfg["picks"][cfg["pair"]]]
    fig = make_figure(clips, margins, cfg)
    spec = ", ".join(f"{cfg['words'][cfg['pair']][cfg['pair'].index(c)]} {cond} q{q}" for c, cond, q in cfg["picks"][cfg["pair"]])
    fig.suptitle(f"UMT L9 #{cfg['feature']}, MLM run, pair {a}/{b}. Clips (class, picked by activation under, quantile): {spec}",
                 fontsize=9, x=0.15, ha="left")
    out = cfg["out_dir"] / f"frames_{a}_{b}.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
