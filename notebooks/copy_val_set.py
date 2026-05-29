"""
Copies SSv2 validation clips into data/ssv2_val_set/ for transfer to Isambard.

Reads the validation JSON for clip IDs, copies matching .webm files.
Run once locally, then rsync data/ssv2_val_set/ up.
"""

import json
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).parent.parent

SOURCE_DIR = ROOT / "data" / "ssv2" / "20bn-something-something-v2"
DEST_DIR   = ROOT / "data" / "ssv2_val_set"
VAL_JSON   = ROOT / "data" / "ssv2" / "labels" / "validation.json"


def main() -> None:
    with open(VAL_JSON) as f:
        clips = json.load(f)

    clip_ids = [str(c["id"]) for c in clips]
    print(f"{len(clip_ids):,} clips in validation set")

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for clip_id in tqdm(clip_ids, desc="Copying"):
        src = SOURCE_DIR / f"{clip_id}.webm"
        dst = DEST_DIR   / f"{clip_id}.webm"

        if dst.exists():
            continue
        if not src.exists():
            missing.append(clip_id)
            continue

        shutil.copy2(src, dst)

    copied = len(list(DEST_DIR.glob("*.webm")))
    print(f"\n{copied:,} clips in {DEST_DIR}")
    if missing:
        print(f"Warning: {len(missing)} clips not found in source (e.g. {missing[0]})")


if __name__ == "__main__":
    main()
