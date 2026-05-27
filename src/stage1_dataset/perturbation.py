"""
Perturbation pipeline — model-agnostic.

Reads source WebM clips and writes two perturbed variants per clip:
  B — first/last: first half of frames → frame 0, remainder → last frame
  C — shuffle: frames randomly permuted (seed derived from clip_id, never stored)

Output layout relative to output_dir:
  output_dir/B/{clip_id}B.webm
  output_dir/C/{clip_id}C.webm
  output_dir.parent/perturbation_metadata.parquet
"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pandas as pd
from tqdm import tqdm

PIXEL_FMT = "yuv420p"
_CODEC_ENCODER: dict[str, str] = {"vp8": "libvpx", "vp9": "libvpx-vp9"}


def get_clip_properties(clip_path: Path) -> dict:
    with av.open(str(clip_path)) as container:
        s = container.streams.video[0]
        return {
            "frame_rate": Fraction(s.average_rate),
            "width": s.width & ~1,    # yuv420p requires even dimensions
            "height": s.height & ~1,
            "n_frames": s.frames or 0,  # WebM headers often omit frame count; updated by process_clip
            "codec": s.codec_context.name,
            "clip_id": clip_path.stem,
        }


def read_frames(clip_path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(clip_path)) as container:
        for frame in container.decode(video=0):
            w = frame.width & ~1
            h = frame.height & ~1
            frames.append(frame.reformat(width=w, height=h, format=PIXEL_FMT).to_ndarray())
    return frames


def apply_first_last(frames: list[np.ndarray]) -> list[np.ndarray]:
    n = len(frames)
    half = n // 2
    return [frames[0]] * half + [frames[-1]] * (n - half)


def apply_shuffle(frames: list[np.ndarray], seed: int) -> list[np.ndarray]:
    indices = np.random.default_rng(seed).permutation(len(frames))
    return [frames[i] for i in indices]


def write_clip(frames: list[np.ndarray], output_path: Path, properties: dict) -> None:
    encoder = _CODEC_ENCODER.get(properties["codec"], "libvpx")
    with av.open(str(output_path), mode="w", format="webm") as output:
        stream = output.add_stream(encoder, rate=properties["frame_rate"])
        stream.width = properties["width"]
        stream.height = properties["height"]
        stream.pix_fmt = PIXEL_FMT
        stream.codec_context.gop_size = 1  # every frame is a keyframe

        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format=PIXEL_FMT)
            for packet in stream.encode(frame):
                output.mux(packet)

        for packet in stream.encode():  # flush encoder
            output.mux(packet)


def process_clip(clip_path: Path, output_dir: Path) -> dict:
    props = get_clip_properties(clip_path)
    clip_id = props["clip_id"]

    frames = read_frames(clip_path)
    props["n_frames"] = len(frames)

    dir_B = output_dir / "B"
    dir_C = output_dir / "C"
    dir_B.mkdir(parents=True, exist_ok=True)
    dir_C.mkdir(parents=True, exist_ok=True)

    path_B = dir_B / f"{clip_id}B.webm"
    path_C = dir_C / f"{clip_id}C.webm"

    write_clip(apply_first_last(frames), path_B, props)
    write_clip(apply_shuffle(frames, int(clip_id) % 2**32), path_C, props)

    return {
        "clip_id": clip_id,
        "source_path": str(clip_path),
        "n_frames": props["n_frames"],
        "frame_rate": str(props["frame_rate"]),
        "width": props["width"],
        "height": props["height"],
        "codec": props["codec"],
        "path_B": str(path_B),
        "path_C": str(path_C),
    }


def process_dataset(clip_list: list[Path], output_dir: Path) -> None:
    records = [
        process_clip(clip_path, output_dir)
        for clip_path in tqdm(clip_list, desc="Perturbing clips")
    ]
    parquet_path = output_dir.parent / "perturbation_metadata.parquet"
    pd.DataFrame(records).to_parquet(parquet_path, index=False)
