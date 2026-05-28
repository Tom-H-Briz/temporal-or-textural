"""
Rebuilds perturbation_metadata.parquet from existing B/C clips on disk.
Use when clips are complete but the parquet is missing or partial.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from tqdm import tqdm
from stage1_dataset.perturbation import get_clip_properties

VIDEO_DIR = ROOT / "data" / "ssv2" / "20bn-something-something-v2"
PERTURBED_DIR = ROOT / "data" / "perturbed"
PARQUET_PATH = ROOT / "data" / "perturbation_metadata.parquet"


def main() -> None:
    b_clips = sorted((PERTURBED_DIR / "B").glob("*.webm"))
    print(f"Found {len(b_clips):,} clips in B/")

    records = []
    missing_source = []
    missing_c = []

    for path_B in tqdm(b_clips, desc="Rebuilding metadata"):
        clip_id = path_B.stem[:-1]  # strip trailing "B"
        path_C = PERTURBED_DIR / "C" / f"{clip_id}C.webm"
        source = VIDEO_DIR / f"{clip_id}.webm"

        if not path_C.exists():
            missing_c.append(clip_id)
            continue
        if not source.exists():
            missing_source.append(clip_id)
            continue

        props = get_clip_properties(source)

        records.append({
            "clip_id": clip_id,
            "source_path": str(source),
            "n_frames": props["n_frames"],
            "frame_rate": str(props["frame_rate"]),
            "width": props["width"],
            "height": props["height"],
            "codec": props["codec"],
            "path_B": str(path_B),
            "path_C": str(path_C),
        })

    if missing_c:
        print(f"  Warning: {len(missing_c)} clips missing C variant (skipped)")
    if missing_source:
        print(f"  Warning: {len(missing_source)} clips missing source (skipped)")

    pd.DataFrame(records).to_parquet(PARQUET_PATH, index=False)
    print(f"Written {len(records):,} records → {PARQUET_PATH}")


if __name__ == "__main__":
    main()
