"""
Quick baseline-only accuracy check on a random clip sample — calls the exact same
run_spliced_accuracy pipeline used for the real spliced-accuracy runs (baseline_only
skips SAE/dim_mean loading and the splice pass), so this number is directly
comparable rather than a separately-reimplemented approximation of it.

Usage:
    uv run python notebooks/check_kinetics_baseline.py --n-clips 3000
"""

import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import DATASET_REGISTRY
from spliced_accuracy_vm import run_spliced_accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="kinetics400")
    parser.add_argument("--n-clips", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    video_dir = Path(os.environ.get("VIDEO_DIR") or DATASET_REGISTRY[args.dataset_name]["video_dir"])
    all_clips = sorted(p.name for ext in ("*.mp4", "*.webm", "*.avi") for p in video_dir.glob(ext))
    rng = random.Random(args.seed)
    eval_clips = rng.sample(all_clips, min(args.n_clips, len(all_clips)))
    print(f"Sampled {len(eval_clips):,} of {len(all_clips):,} clips (seed={args.seed}) — not the "
          f"held-out validation split, a separate random draw for a quick pre-run check")

    run_spliced_accuracy(
        model_name="videomae", dataset_name=args.dataset_name,
        eval_clips=eval_clips, baseline_only=True,
    )


if __name__ == "__main__":
    main()
