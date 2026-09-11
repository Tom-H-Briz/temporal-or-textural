"""
Known-clip smoke test for the ViViT Kinetics-400 label mapping — run BEFORE the
full perturbation-accuracy job.

google/vivit-b-16x2-kinetics400 ships generic LABEL_N id2label, so label2id comes
from resolve_k400_label2id's canonical alphabetical fallback. This script runs a
handful of val.csv clips with known ground-truth classes through the model: with
the right mapping most predictions name the true class (or a close neighbour);
with a wrong mapping predictions are effectively random (~1/400 per clip). Enough
signal to kill a bad mapping before burning the 8h accuracy job.

Local data/kinetics400/val is empty — run on Isambard:
    VIDEO_DIR=/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset \
    python notebooks/check_vivit_label_map.py
"""

import csv
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import (
    CHECKPOINT_REGISTRY, DATASET_REGISTRY, FRAME_SAMPLERS, MODEL_REGISTRY,
    resolve_k400_label2id,
)

CFG = {
    "model_name":  "vivit",
    "n_clips":     6,  # distinct ground-truth classes, one clip each
    "labels_csv":  os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data/kinetics400/annotations/val.csv")
    ),
}


def pick_clips(n_clips: int) -> list[dict]:
    """First n_clips val.csv rows with a clip on disk, all distinct labels."""
    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY["kinetics400"]["video_dir"])
    picked, seen_labels = [], set()
    with open(CFG["labels_csv"]) as f:
        for row in csv.DictReader(f):
            start, end = int(float(row["time_start"])), int(float(row["time_end"]))
            candidates = (list(video_dir.glob(f"{row['youtube_id']}_{start:06d}_{end:06d}.*"))
                          or list(video_dir.glob(f"{row['youtube_id']}.*")))
            if not candidates or row["label"] in seen_labels:
                continue
            picked.append({"path": candidates[0], "label": row["label"]})
            seen_labels.add(row["label"])
            if len(picked) == n_clips:
                break
    if len(picked) < n_clips:
        raise RuntimeError(f"Only found {len(picked)} clips on disk — VIDEO_DIR set correctly?")
    return picked


def main() -> None:
    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], "kinetics400")]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint).eval()

    label2id = resolve_k400_label2id(CFG["model_name"])
    id2name  = {v: k for k, v in label2id.items()}
    sampler  = FRAME_SAMPLERS["kinetics400"]

    import av
    n_correct = 0
    for clip in pick_clips(CFG["n_clips"]):
        container = av.open(str(clip["path"]))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        sampled = [frames[i] for i in sampler(len(frames), model_cfg["num_frames"])]
        pixel_values = processor(sampled, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            logits = model(pixel_values=pixel_values).logits[0]
        pred = logits.argmax().item()
        n_correct += int(pred == label2id[clip["label"]])
        print(f"  true={clip['label']!r:<40} pred={id2name[pred]!r:<40} "
              f"prob={torch.softmax(logits, -1)[pred]:.3f}")


    print(f"\n{n_correct}/{CFG['n_clips']} correct — "
          + ("label map consistent; safe to launch the accuracy job" if n_correct >= CFG["n_clips"] // 2
             else "SUSPECT: wrong-mapping predictions look random — do not launch, revisit the fallback"))


if __name__ == "__main__":
    main()
