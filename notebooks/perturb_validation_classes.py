"""
Runs the perturbation pipeline on a fixed subset of SSv2 validation clips.

Target classes (5):
  Pushing something from right to left
  Pushing something from left to right
  Tearing something just a little bit
  Tearing something into two pieces
  Pushing something so that it spins around

Expected: 1505 clips → data/perturbed/{B,C}/ + data/perturbation_metadata.parquet
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))   # notebooks/ → class_selection importable
sys.path.insert(0, str(ROOT / "src"))             # src/ → stage1_dataset importable

from class_selection import CFG, _strip_brackets, load_metadata
from stage1_dataset.perturbation import process_dataset

TARGET_CLASSES = {
    "Pushing something from right to left",
    "Pushing something from left to right",
    "Tearing something just a little bit",
    "Tearing something into two pieces",
    "Pushing something so it spins",
}

OUTPUT_DIR = ROOT / "data" / "perturbed"
EXPECTED_CLIPS = 1505


def main() -> None:
    print("Loading metadata...")
    label_map, clips, _ = load_metadata(CFG["labels_path"], CFG["validation_path"])

    selected = [c for c in clips if _strip_brackets(c["template"]) in TARGET_CLASSES]
    print(f"  {len(selected):,} clips from {len(TARGET_CLASSES)} classes")
    if len(selected) != EXPECTED_CLIPS:
        print(f"  Warning: expected {EXPECTED_CLIPS}, got {len(selected)}")

    video_dir = Path(CFG["video_dir"])
    clip_paths = [video_dir / f"{c['id']}.webm" for c in selected]

    missing = [p for p in clip_paths if not p.exists()]
    if missing:
        print(f"  Warning: {len(missing)} clip(s) missing from disk, e.g. {missing[0]}")
    clip_paths = [p for p in clip_paths if p.exists()]

    print(f"Processing {len(clip_paths):,} clips → {OUTPUT_DIR}")
    process_dataset(clip_paths, OUTPUT_DIR)

    parquet_path = OUTPUT_DIR.parent / "perturbation_metadata.parquet"
    print(f"Done. Parquet: {parquet_path}")


if __name__ == "__main__":
    main()
