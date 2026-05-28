"""
Shared utilities for temporal-or-textural notebooks.
"""

from pathlib import Path

import av
import torch
from torch.utils.data import Dataset


class SSv2ClipDataset(Dataset):
    """
    Loads SSv2 WebM clips from explicit paths.

    With labels: __getitem__ returns (pixel_values, label) — for classification.
    Without labels: __getitem__ returns pixel_values — for SAE training.
    """

    def __init__(
        self,
        clip_paths: list[Path],
        processor,
        num_frames: int,
        labels: list[int] | None = None,
    ) -> None:
        assert labels is None or len(labels) == len(clip_paths)
        self.clip_paths = clip_paths
        self.processor = processor
        self.num_frames = num_frames
        self.labels = labels

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int):
        path = self.clip_paths[idx]
        container = av.open(str(path))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()

        n = len(frames)
        indices = torch.linspace(0, n - 1, self.num_frames).long().tolist()
        sampled = [frames[i] for i in indices]

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)

        if self.labels is not None:
            return pixel_values, self.labels[idx]
        return pixel_values
