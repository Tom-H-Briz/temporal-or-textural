"""
Shared utilities for temporal-or-textural notebooks.
"""

import json
from pathlib import Path

import av
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification

MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"
NUM_FRAMES = 16
NUM_CLASSES = 174


def _strip_brackets(template: str) -> str:
    return template.replace("[", "").replace("]", "")


def load_metadata(
    labels_path: str, validation_path: str
) -> tuple[dict[str, int], list[dict], dict[int, str]]:
    with open(labels_path) as f:
        raw: dict[str, str] = json.load(f)

    first_key = next(iter(raw))
    if first_key.isdigit():
        label_map = {name: int(idx) for idx, name in raw.items()}
        id2template = {int(idx): name for idx, name in raw.items()}
    else:
        label_map = {template: int(idx) for template, idx in raw.items()}
        id2template = {int(idx): template for template, idx in raw.items()}

    with open(validation_path) as f:
        all_clips: list[dict] = json.load(f)

    clips = [c for c in all_clips if _strip_brackets(c["template"]) in label_map]
    n_dropped = len(all_clips) - len(clips)
    if n_dropped:
        print(f"  Warning: dropped {n_dropped} clips with templates not in labels.json")

    return label_map, clips, id2template


def make_sae_splice_hook(sae: torch.nn.Module, dim_mean: torch.Tensor):
    """
    Returns a forward hook that replaces a layer's output with its SAE reconstruction.
    Register on the target encoder layer; remove the handle when done.
    SAE must be in eval mode with running_threshold initialised before use.
    """
    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        B, T, D = hidden.shape
        tokens = (hidden.reshape(B * T, D) - dim_mean).float()
        with torch.no_grad():
            _, _, x_hat = sae(tokens)
        reconstructed = (x_hat + dim_mean).reshape(B, T, D).to(hidden.dtype)
        if isinstance(output, tuple):
            return (reconstructed,) + output[1:]
        return reconstructed
    return _hook


def run_inference(
    model, dataloader: DataLoader, device: str  # model: any HF video classifier with .logits
) -> tuple[list[int], list[int]]:
    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()

    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc="Inference"):
            pixel_values = pixel_values.to(device)
            preds = model(pixel_values=pixel_values).logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


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
