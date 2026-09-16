"""Does C1's tubelet-pair shuffle change ViViT's raw logits at all?

Compares full R vs C1 logit vectors (not just argmax/accuracy) on a handful of
K400 clips. Bit-identical logits => the shuffle isn't reaching the model (bug).
Slightly different logits that rarely flip argmax => genuine near-null effect
(K400 clips are near-static within the sampled window, so reordering already-
similar tubelets barely moves the decision).

Usage: uv run python notebooks/diagnose_c1_logit_effect.py
"""
import os
import sys
import zlib
from pathlib import Path

import av
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(Path(__file__).parent))

from perturb_accuracy_vm import apply_shuffle_pairs
from spliced_accuracy_vm import load_kinetics_metadata
from ToT_utils import (
    CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY,
    get_frame_sampler, get_processor, resolve_k400_label2id,
)

CFG = {
    "model_name":   "vivit",
    "dataset_name": "kinetics400",
    "n_clips":      10,
    "labels_csv":   os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data/kinetics400/annotations/val.csv")
    ),
}


def load_frames(path: Path) -> list:
    container = av.open(str(path))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    return frames


def main() -> None:
    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], CFG["dataset_name"])]
    processor  = get_processor(model_cfg, checkpoint)  # do_normalize=False, matches the real run
    model      = model_cfg["model_class"].from_pretrained(checkpoint).eval()
    frame_sampler = get_frame_sampler(CFG["dataset_name"], model_cfg)

    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[CFG["dataset_name"]]["video_dir"])
    label2id = resolve_k400_label2id(CFG["model_name"])
    clip_paths, labels, id2label = load_kinetics_metadata(CFG["labels_csv"], video_dir, label2id)

    for idx in range(CFG["n_clips"]):
        frames = load_frames(clip_paths[idx])
        sampled = [frames[i] for i in frame_sampler(len(frames), model_cfg["num_frames"])]
        seed = zlib.crc32(clip_paths[idx].stem.encode())  # same seed convention as the real run
        shuffled = apply_shuffle_pairs(sampled, seed)

        with torch.no_grad():
            r_logits  = model(pixel_values=processor(sampled,   return_tensors="pt")["pixel_values"]).logits[0]
            c1_logits = model(pixel_values=processor(shuffled, return_tensors="pt")["pixel_values"]).logits[0]

        max_abs_diff = (r_logits - c1_logits).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(r_logits, c1_logits, dim=0).item()
        r_pred, c1_pred = r_logits.argmax().item(), c1_logits.argmax().item()
        print(f"{clip_paths[idx].stem:>15}  label={id2label[labels[idx]]!r:35}  "
              f"R={id2label[r_pred]!r:25} C1={id2label[c1_pred]!r:25}  "
              f"max|dlogit|={max_abs_diff:.4f}  cos_sim={cos_sim:.6f}")


if __name__ == "__main__":
    main()
