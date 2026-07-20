"""
Read-only diagnostic: reports what's actually staged for Kinetics-400 on Isambard,
so spliced_accuracy_vm.py's K400 label loading (load_kinetics_metadata — assumed
DeepMind CSV columns + {youtube_id}_{start:06d}_{end:06d}.ext filenames) can be
specced against real data instead of assumptions. No GPU, no model load.

Stdlib only, deliberately — ToT_utils.py imports torch/transformers/av at module
level, which fail to install on Isambard's login node (no GPU there); this script
needs to run on the login node, where /scratch is visible, so it can't depend on it.

Usage:
    python3 notebooks/inspect_kinetics_data.py
    VIDEO_DIR=/scratch/... KINETICS_LABELS_CSV=/path/to/val.csv python3 notebooks/inspect_kinetics_data.py
"""

import csv
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Mirrors DATASET_REGISTRY["kinetics400"]["video_dir"] without importing ToT_utils.
DEFAULT_VIDEO_DIR = ROOT / "data" / "kinetics400" / "val"

CFG = {
    "video_dir":  os.environ.get("VIDEO_DIR") or str(DEFAULT_VIDEO_DIR),
    "labels_csv": os.environ.get("KINETICS_LABELS_CSV")
                  or str(ROOT / "data" / "kinetics400" / "annotations" / "val.csv"),
}


def probe_video_dir(video_dir: Path) -> None:
    print(f"\n=== video_dir: {video_dir} ===")
    if not video_dir.exists():
        print("  DOES NOT EXIST")
        return
    entries = list(video_dir.iterdir())
    dirs, files = [e for e in entries if e.is_dir()], [e for e in entries if e.is_file()]
    print(f"  {len(dirs)} subdirectories, {len(files)} files directly inside")
    if dirs:
        print(f"  Sample subdirs (possible class-per-dir layout): {[d.name for d in dirs[:10]]}")
    print(f"  Top-level file extensions: {dict(Counter(f.suffix for f in files))}")


def probe_recursive(video_dir: Path) -> None:
    if not video_dir.exists():
        return
    clip_files = [f for f in video_dir.rglob("*")
                  if f.is_file() and f.suffix.lower() in (".mp4", ".webm", ".avi")]
    print(f"\n=== recursive clip count under {video_dir}: {len(clip_files):,} "
          f"(brief claims 19,881) ===")
    print(f"  Sample filenames: {[f.name for f in clip_files[:10]]}")
    if any(f.parent != video_dir for f in clip_files[:5]):
        print(f"  Sample relative paths: {[str(f.relative_to(video_dir)) for f in clip_files[:5]]}")


def probe_labels_csv(labels_csv: Path) -> None:
    print(f"\n=== labels_csv: {labels_csv} ===")
    if not labels_csv.exists():
        print("  DOES NOT EXIST — searching data/kinetics400 for any *.csv...")
        k400_dir = ROOT / "data" / "kinetics400"
        found = list(k400_dir.rglob("*.csv")) if k400_dir.exists() else []
        print(f"  Found: {[str(p) for p in found]}")
        return
    with open(labels_csv) as f:
        header = f.readline().strip()
        rows = f.readlines()
    print(f"  Header: {header}")
    print(f"  Row count: {len(rows):,}")
    print(f"  Sample rows: {[r.strip() for r in rows[:3]]}")
    with open(labels_csv) as f:
        splits = Counter(row.get("split", "?") for row in csv.DictReader(f))
    print(f"  'split' column value counts: {dict(splits)}")


def probe_filename_matching(labels_csv: Path, video_dir: Path, n: int = 5) -> None:
    if not labels_csv.exists() or not video_dir.exists():
        return
    print(f"\n=== filename-matching check (first {n} CSV rows vs assumed "
          f"{{youtube_id}}_{{start:06d}}_{{end:06d}}.ext pattern) ===")
    with open(labels_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            yid = row.get("youtube_id", "?")
            try:
                start, end = int(float(row["time_start"])), int(float(row["time_end"]))
                pattern = f"{yid}_{start:06d}_{end:06d}.*"
                matches = list(video_dir.glob(pattern)) or list(video_dir.rglob(pattern))
            except (KeyError, ValueError):
                pattern, matches = "N/A (missing time_start/time_end column)", []
            print(f"  label={row.get('label')!r} yid={yid} pattern={pattern!r} "
                  f"matches={[m.name for m in matches]}")


def main() -> None:
    print(f"ROOT: {ROOT}")
    print(f"video_dir default (mirrors DATASET_REGISTRY): {DEFAULT_VIDEO_DIR}")
    video_dir, labels_csv = Path(CFG["video_dir"]), Path(CFG["labels_csv"])
    probe_video_dir(video_dir)
    probe_recursive(video_dir)
    probe_labels_csv(labels_csv)
    probe_filename_matching(labels_csv, video_dir)


if __name__ == "__main__":
    main()
