"""
Perturbation A — single midpoint frame.

All frames replaced with the clip's midpoint frame (index n // 2),
the same split boundary used in condition B.

Output layout:
  data/perturbed/A/{clip_id}A.webm
  Adds path_A column to data/perturbation_metadata.parquet
"""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from perturbation import get_clip_properties, read_frames, write_clip

CFG = {
    "metadata_path": str(ROOT / "data" / "perturbation_metadata.parquet"),
    "output_dir": str(ROOT / "data" / "perturbed"),
}


def apply_midpoint_frame(frames: list) -> list:
    """All frames replaced with the midpoint frame (index n // 2)."""
    return [frames[len(frames) // 2]] * len(frames)


def process_clip_A(source_path: Path, output_dir: Path) -> str:
    props = get_clip_properties(source_path)
    clip_id = props["clip_id"]

    frames = read_frames(source_path)
    props["n_frames"] = len(frames)

    dir_A = output_dir / "A"
    dir_A.mkdir(parents=True, exist_ok=True)
    path_A = dir_A / f"{clip_id}A.webm"

    write_clip(apply_midpoint_frame(frames), path_A, props)
    return str(path_A)


def main() -> None:
    meta = pd.read_parquet(CFG["metadata_path"])
    output_dir = Path(CFG["output_dir"])

    paths_A = [
        process_clip_A(Path(row["source_path"]), output_dir)
        for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Condition A")
    ]

    meta["path_A"] = paths_A
    meta.to_parquet(CFG["metadata_path"], index=False)
    print(f"Updated → {CFG['metadata_path']}")
    print(f"Wrote {len(meta)} clips → {output_dir / 'A'}")


if __name__ == "__main__":
    main()
