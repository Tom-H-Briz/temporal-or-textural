"""Frame figure for one L9 SAE feature in the UMT MLM runs: R / C / Rev rows per clip (CPU, local).

Overlay = the feature's activation, signed by its MLM contribution (red = toward the clip's true word).
The C row shows the shuffled frames in the order the model saw them (native-frame shuffle, seed = clip id,
then UMT 'middle' sampling, as PerturbedKineticsDataset), each titled with its native time t (0 = start).
Row labels give the MLM answer and margin T_ab. Works for any feature and any pair in an MLM contrast
run (umt_mlm_feature_profile.CFG["in_dirs"]); pair words are decoded from the run's token ids.
Clips (env, first one set wins):
  CLIPS=id,id,...           explicit clip ids (class looked up)
  PICKS=cls:cond:q,...      class, condition it is ranked under, activation quantile (1.0 = strongest)
  (neither)                 the class it fires on most: R q1.0 / 0.5 / 0.25; the other class: Rev q1.0, C q1.0
CONDS=R,C,Rev picks the rows; TAG=name adds a suffix to the output file.
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
sys.path[:0] = [str(ROOT / "notebooks"), str(Path(__file__).parent), str(ROOT / "src/stage3_analysis")]
from umt_pair_feature_tracks import draw_row, feature_maps   # shared overlay / row drawing
from umt_wrapper import sample_frames_umt
import umt_mlm_feature_profile as prof_mod

CFG = {
    "feature":   int(os.environ.get("FEATURE", "4250")),
    "pair":      tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(",")),
    "clips":     os.environ.get("CLIPS", ""),
    "picks":     os.environ.get("PICKS", ""),
    "conds":     tuple(os.environ.get("CONDS", "R,C,Rev").split(",")),
    "tag":       os.environ.get("TAG", ""),
    "video_dir": ROOT / "data/ssv2_val_set",
    "n_frames": 12, "grid": 14, "img": 224, "track_axis": "row",
}
CFG["in_dir"] = prof_mod.find_run(CFG["pair"], prof_mod.CFG["in_dirs"], "per_clip.parquet")  # sparse codes
CFG["out_dir"] = prof_mod.profile_dir(CFG["feature"])


def frames_rc(cid, cfg):
    """12 R frames, 12 C frames (as PerturbedKineticsDataset: permute all native frames with
    seed = clip id, then sample), and each C frame's native time in [0, 1]."""
    with av.open(str(cfg["video_dir"] / f"{cid}.webm")) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    n, idx = len(fr), sample_frames_umt(len(fr), cfg["n_frames"])
    perm = np.random.default_rng(int(cid)).permutation(n)      # = perturbation.apply_shuffle
    sq = lambda a: np.asarray(Image.fromarray(a).resize((cfg["img"],) * 2, Image.BICUBIC))
    return [sq(fr[i]) for i in idx], [sq(fr[perm[i]]) for i in idx], perm[idx] / (n - 1)


def pair_words(cfg):
    """{class: word} for the pair, decoded from the contrast run's word_a_id / word_b_id."""
    from transformers import BertTokenizer
    pc = pd.read_parquet(cfg["in_dir"] / "per_clip.parquet", columns=["a", "b", "word_a_id", "word_b_id"])
    r = pc[(pc.a == cfg["pair"][0]) & (pc.b == cfg["pair"][1])].iloc[0]
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    return dict(zip(cfg["pair"], tok.convert_ids_to_tokens([int(r.word_a_id), int(r.word_b_id)])))


def pick(prof, cls, cond, q):
    """Clip id of class cls at activation quantile q under cond (1.0 = strongest)."""
    d = prof[(prof.true_class == cls) & (prof.condition == cond)].sort_values("z")
    return int(d.clip_id.iloc[min(int(q * len(d)), len(d) - 1)])


def default_picks(prof, cfg):
    """The class the feature fires on most (R): strongest / median / 25th pct; other class: Rev, C."""
    on = prof[prof.condition == "R"].groupby("true_class").z.mean().idxmax()
    off = cfg["pair"][1 - cfg["pair"].index(on)]
    return [(on, "R", 1.0), (on, "R", 0.5), (on, "R", 0.25), (off, "Rev", 1.0), (off, "C", 1.0)]


def choose_clips(prof, cfg):
    """[(clip_id, class, how it was chosen)] from CLIPS, else PICKS, else the defaults."""
    if cfg["clips"]:
        cls = prof.drop_duplicates("clip_id").set_index("clip_id").true_class
        return [(int(c), int(cls[int(c)]), "chosen") for c in cfg["clips"].split(",")]
    picks = [(int(c), cond, float(q)) for c, cond, q in (p.split(":") for p in cfg["picks"].split(","))] \
        if cfg["picks"] else default_picks(prof, cfg)
    return [(pick(prof, c, cond, q), c, f"{cond} q{q:g}") for c, cond, q in picks]


def clip_block(axes, cid, cls, how, margins, norm, cfg):
    """One row per condition in CONDS: R, C (native time over each frame), Rev (R frames reversed)."""
    fr, fc, t_nat = frames_rc(cid, cfg)
    words, (a, b) = cfg["words"], cfg["pair"]
    frames_of = {"R": fr, "C": fc, "Rev": fr[::-1]}
    for row, cond in enumerate(cfg["conds"]):
        frames = frames_of[cond]
        maps, sign = feature_maps(cid, cond, cfg["feature"], cls, cfg)
        T = margins.loc[(cid, cond)]
        label = f"{cond}: MLM says '{words[a] if T > 0 else words[b]}' ({T:+.1f})"
        if row == 0:
            label = f"{words[cls].upper()} clip {cid} [{how}]\n" + label
        draw_row(axes[row], frames, maps, sign, norm, label, cfg)
        if cond == "C":
            for ax, tn in zip(axes[row], t_nat):
                ax.set_title(f"t={tn:.2f}", fontsize=6, pad=1)


def make_figure(clips, margins, cfg):
    """len(CONDS) rows per clip, one symmetric colour scale over every map shown."""
    k = len(cfg["conds"])
    maps = [feature_maps(c, cond, cfg["feature"], cls, cfg)[0] for c, cls, _ in clips for cond in cfg["conds"]]
    vmax = float(np.percentile(np.stack(maps), 99.5)) or 1.0
    norm = mcolors.Normalize(-vmax, vmax)
    fig, axes = plt.subplots(k * len(clips), cfg["n_frames"], figsize=(15, 1.35 * k * len(clips)), squeeze=False)
    for i, (cid, cls, how) in enumerate(clips):
        clip_block(axes[k * i:k * i + k], cid, cls, how, margins, norm, cfg)
    fig.subplots_adjust(left=0.17, right=0.95, top=1 - 0.5 / (k * len(clips)), bottom=0.01, wspace=0.03, hspace=0.18)
    cax = fig.add_axes([0.962, 0.35, 0.008, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=norm), cax=cax)
    cb.set_label(f"#{cfg['feature']} activation\nred = toward the clip's true word", fontsize=8)
    return fig


def main():
    cfg = dict(CFG)
    a, b = cfg["pair"]
    cfg["words"] = pair_words(cfg)
    prof = prof_mod.load_profile(cfg["feature"])
    prof = prof[(prof.a == a) & (prof.b == b)]
    margins = prof.set_index(["clip_id", "condition"]).T_ab
    clips = choose_clips(prof, cfg)
    fig = make_figure(clips, margins, cfg)
    fig.suptitle(f"UMT L9 #{cfg['feature']}, MLM run '{cfg['in_dir'].name}', {cfg['words'][a]} ({a}) vs "
                 f"{cfg['words'][b]} ({b}). [cond qX] = clip at activation quantile X under that condition",
                 fontsize=9, x=0.17, ha="left")
    out = cfg["out_dir"] / f"frames_{a}_{b}{'_' + cfg['tag'] if cfg['tag'] else ''}.png"
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
