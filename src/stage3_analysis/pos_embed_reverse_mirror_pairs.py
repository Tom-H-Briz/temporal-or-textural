"""
Does position-embedding reversal FLIP mirror-pair predictions?

Targeted probe on SSv2 direction-defined class pairs (left/right push & pull,
camera left/right, closer/away). The SL-subset sample contains none of the
left/right classes, so this samples fresh clips per class from validation.json
and runs three passes per clip: baseline, reverse [7..0], and a random
non-identity temporal shuffle (the contrast — generic position damage should
scatter predictions, not flip them to the mirror).

Readout: P(reversal predicts the mirror class | true class, baseline-correct),
per direction of each pair, plus the same under shuffle.

Usage:
    uv run python src/stage3_analysis/pos_embed_reverse_mirror_pairs.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.dfa_engine import _preprocess_clip
from stage3_analysis.pos_embed_shuffle_accuracy import (
    apply_temporal_perm, draw_perm, forward_stats,
)
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_metadata, _strip_brackets

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "n_per_class": 40,
    "seed": 0,
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "T": 8, "SPATIAL": 196,
    "reverse": [7, 6, 5, 4, 3, 2, 1, 0],
    "pairs": [
        ("Pulling something from left to right", "Pulling something from right to left"),
        ("Pushing something from left to right", "Pushing something from right to left"),
        ("Turning the camera left while filming something",
         "Turning the camera right while filming something"),
        ("Moving something closer to something", "Moving something away from something"),
        ("Moving something and something closer to each other",
         "Moving something and something away from each other"),
    ],
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "validation_path": ROOT / "data/ssv2/labels/validation.json",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "out_path": ROOT / "outputs/analysis/pos_embed_shift/mirror_pairs_reverse.csv",
}


def sample_clips(cfg: dict) -> pd.DataFrame:
    """Up to n_per_class validation clips per mirror-pair member, seeded."""
    label2id, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    wanted = {t for pair in cfg["pairs"] for t in pair}
    by_class = {}
    for c in clips:
        t = _strip_brackets(c["template"])
        if t in wanted and (cfg["video_dir"] / f"{c['id']}.webm").exists():
            by_class.setdefault(t, []).append((c["id"], label2id[t]))
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for t, pool_ in by_class.items():
        pick = rng.choice(len(pool_), size=min(cfg["n_per_class"], len(pool_)),
                          replace=False)
        rows += [{"clip_id": pool_[i][0], "class_id": pool_[i][1], "template": t}
                 for i in pick]
    return pd.DataFrame(rows)


def main() -> None:
    cfg = CFG
    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], cfg["dataset"])]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(cfg["device"]).eval()
    model.requires_grad_(False)
    pos = model.videomae.embeddings.position_embeddings
    original = pos.data.clone()

    clips = sample_clips(cfg)
    print(f"Device: {cfg['device']}  {len(clips)} clips, "
          f"{clips.template.nunique()} classes, 3 passes each")
    rows = []
    for i, (_, r) in enumerate(clips.iterrows()):
        pv = _preprocess_clip(cfg["video_dir"] / f"{r.clip_id}.webm",
                              model_cfg["num_frames"], processor, cfg["device"])
        _, base_pred = forward_stats(model, pv, int(r.class_id))
        for arm, assign in [("reverse", cfg["reverse"]),
                            ("shuffle", draw_perm(cfg, r.clip_id))]:
            apply_temporal_perm(model, assign, cfg)
            _, pred = forward_stats(model, pv, int(r.class_id))
            pos.data = original
            rows.append({"clip_id": r.clip_id, "class_id": int(r.class_id),
                         "template": r.template, "arm": arm,
                         "baseline_pred": base_pred, "baseline_correct":
                         base_pred == int(r.class_id),
                         "pred": pred, "correct": pred == int(r.class_id)})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(clips)}]")
    assert torch.equal(pos.data, original), "embeddings not restored"
    summarise(pd.DataFrame(rows), cfg)


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    label2id, _, id2label = load_metadata(cfg["labels_path"], cfg["validation_path"])
    out = []
    for a_t, b_t in cfg["pairs"]:
        ia, ib = label2id[a_t], label2id[b_t]
        for true_id, mirror_id in [(ia, ib), (ib, ia)]:
            for arm in ("reverse", "shuffle"):
                g = df[(df.arm == arm) & (df.class_id == true_id)
                       & (df.baseline_correct)]
                out.append({
                    "pair": f"{a_t[:28]} <-> {b_t[:28]}", "true_class": id2label[true_id][:50],
                    "arm": arm, "n_baseline_correct": len(g),
                    "acc": g.correct.mean() if len(g) else float("nan"),
                    "pred_mirror_rate": (g.pred == mirror_id).mean() if len(g) else float("nan")})
    s = pd.DataFrame(out)
    s.to_csv(cfg["out_path"], index=False)
    print(s.to_string(index=False))
    print(f"-> {cfg['out_path']}")


if __name__ == "__main__":
    main()
