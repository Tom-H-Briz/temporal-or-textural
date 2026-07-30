"""
Loads a checkpoint and lets transformers' own from_pretrained load-report print
naturally (the "Key | Status | UNEXPECTED/MISSING" table) — no inference, no GPU
needed, just checking whether any parameters come back missing/unexpected under
the currently-installed transformers version, for a specific model/dataset.

Usage:
    python notebooks/check_model_load_report.py --model-name videomae --dataset-name ssv2
    python notebooks/check_model_load_report.py --model-name timesformer --dataset-name ssv2
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="videomae")
    parser.add_argument("--dataset-name", type=str, default="ssv2")
    args = parser.parse_args()

    model_cfg  = MODEL_REGISTRY[args.model_name]
    checkpoint = CHECKPOINT_REGISTRY[(args.model_name, args.dataset_name)]
    print(f"Loading {checkpoint} ...")
    model_cfg["model_class"].from_pretrained(checkpoint)
    print("Loaded (see report above for missing/unexpected keys).")


if __name__ == "__main__":
    main()
