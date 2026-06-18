"""
Generate probe_config.json — the reproducibility artifact for layer_selector.py.

Resolves train/test/warmup clip splits once and saves clip IDs, class IDs, and
class names to JSON. Run once before layer_selector.py.

Usage: uv run python notebooks/make_probe_config.py
"""

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import _strip_brackets, load_metadata

CFG = {
    "model_name":       "timesformer",
    "layers":           [3, 5, 7, 9, 11],
    "sae_k":            64,
    "sae_expansion":    8,
    "sae_abbrev":       "tf",
    "pooling":          ["mean", "max"],
    "sklearn_C":        1.0,
    "sklearn_max_iter": 1000,
    "sklearn_solver":   "lbfgs",
    "warmup_n":         200,
    "test_frac":        0.2,
    "labels_path":      os.environ.get("LABELS_PATH",     str(ROOT / "data" / "ssv2" / "labels" / "labels.json")),
    "validation_path":  os.environ.get("VALIDATION_PATH", str(ROOT / "data" / "ssv2" / "labels" / "validation.json")),
    "video_dir":        os.environ.get("VIDEO_DIR",        str(ROOT / "data" / "ssv2_val_set")),
    "output_path":      str(ROOT / "outputs" / "sae" / "probe_config.json"),
}


def main() -> None:
    label_map, clips, _ = load_metadata(CFG["labels_path"], CFG["validation_path"])
    video_dir = Path(CFG["video_dir"])

    all_clips = [
        {
            "id":       c["id"],
            "class_id": label_map[_strip_brackets(c["template"])],
            "template": _strip_brackets(c["template"]),
        }
        for c in clips
        if (video_dir / f"{c['id']}.webm").exists()
    ]
    print(f"  {len(all_clips):,} clips on disk")

    rng = random.Random(0)
    rng.shuffle(all_clips)

    warmup_clips = all_clips[: CFG["warmup_n"]]
    remaining    = all_clips[CFG["warmup_n"] :]
    split_idx    = int(len(remaining) * (1 - CFG["test_frac"]))
    train_clips  = remaining[:split_idx]
    test_clips   = remaining[split_idx:]

    print(f"  warmup={len(warmup_clips)}  train={len(train_clips)}  test={len(test_clips)}")

    out = {**CFG, "warmup_clips": warmup_clips, "train_clips": train_clips, "test_clips": test_clips}

    out_path = Path(CFG["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
