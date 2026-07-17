"""
Shared utilities for temporal-or-textural notebooks.
"""

import json
from pathlib import Path

import av
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
)

ROOT = Path(__file__).parent.parent

MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"  # legacy — prefer CHECKPOINT_REGISTRY
NUM_FRAMES = 16   # legacy — prefer MODEL_REGISTRY["videomae"]["num_frames"]
NUM_CLASSES = 174

# Backbone-only fields — none of these vary with dataset. Checkpoint identity is a
# (model x dataset) product, so it lives in CHECKPOINT_REGISTRY instead.
# num_patch_tokens values are literals; the tier-1 shape asserts in setup_model()
# validate them against the model's actual output on every forward pass.
MODEL_REGISTRY: dict[str, dict] = {
    "videomae": {
        "model_class":      VideoMAEForVideoClassification,
        "num_frames":       16,
        "processor_class":  VideoMAEImageProcessor,
        "cls_offset":       0,
        "layer_getter":     lambda model, i: model.videomae.encoder.layer[i],
        "hidden_dim":       768,
        "num_patch_tokens": 1568,
        "position_label":   "tubelet",
    },
    "timesformer": {
        "model_class":      TimesformerForVideoClassification,
        "num_frames":       8,
        "processor_class":  AutoImageProcessor,
        "cls_offset":       1,
        "layer_getter":     lambda model, i: model.timesformer.encoder.layer[i],
        "hidden_dim":       768,
        "num_patch_tokens": 1568,
        "position_label":   "frame",
    },
}

# (model_name, dataset_name) -> HF checkpoint string. The only place a finetuned
# checkpoint string lives — checkpoint identity is a (backbone x dataset) product.
CHECKPOINT_REGISTRY: dict[tuple[str, str], str] = {
    ("videomae", "ssv2"):         "MCG-NJU/videomae-base-finetuned-ssv2",
    ("timesformer", "ssv2"):      "facebook/timesformer-base-finetuned-ssv2",
    ("videomae", "kinetics400"):  "MCG-NJU/videomae-base-finetuned-kinetics",
}

# dataset_name -> backbone-independent dataset paths. labels_path/validation_path
# are None for datasets without SSv2-style template/label JSON metadata — callers
# that need a clip list fall back to globbing video_dir directly in that case.
DATASET_REGISTRY: dict[str, dict] = {
    "ssv2": {
        "data_root":       ROOT / "data" / "ssv2",
        "labels_path":     ROOT / "data" / "ssv2" / "labels" / "labels.json",
        "validation_path": ROOT / "data" / "ssv2" / "labels" / "validation.json",
        "video_dir":       ROOT / "data" / "ssv2_val_set",
    },
    "kinetics400": {
        "data_root":       ROOT / "data" / "kinetics400",
        "labels_path":     None,
        "validation_path": None,
        "video_dir":       ROOT / "data" / "kinetics400" / "val",
    },
}

N_SPATIAL = 196   # 14x14 patch grid — constant across both backbones at this resolution


def gather_by_position(tokens: torch.Tensor, model_flag: str) -> torch.Tensor:
    """
    Group patch tokens (CLS already excluded) by temporal position — VM: tubelet,
    TF: frame — in a canonical position-major axis order for both models, so every
    caller reduces over dim=1 regardless of model_flag.

    VM (videomae) is temporal-major natively: token_idx = position*196 + patch.
    TF (timesformer) is patch-major/frame-minor after the time-embedding permute in
    TimesformerEmbeddings.forward: token_idx = patch*num_frames + position.

    tokens: (num_patch_tokens, ...trailing dims...)
    Returns: (num_positions, N_SPATIAL, ...trailing dims...)
    """
    num_patch_tokens = tokens.shape[0]
    num_positions = num_patch_tokens // N_SPATIAL
    if model_flag == "videomae":
        return tokens.reshape(num_positions, N_SPATIAL, *tokens.shape[1:])
    elif model_flag == "timesformer":
        grouped = tokens.reshape(N_SPATIAL, num_positions, *tokens.shape[1:])
        return grouped.transpose(0, 1)
    else:
        raise ValueError(f"No position-gather rule registered for model_flag={model_flag!r}")


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


def make_sae_splice_hook(
    sae: torch.nn.Module, dim_mean: torch.Tensor, cls_offset: int = 0
):
    """
    Returns a forward hook that replaces a layer's output with its SAE reconstruction.
    Register on the target encoder layer; remove the handle when done.
    SAE must be in eval mode with running_threshold initialised before use.

    cls_offset: 0 for VideoMAE (no CLS token), 1 for TimeSformer. TimeSformer callers
    must pass cls_offset=1 explicitly — the default is VideoMAE-safe only.
    """
    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patch = hidden[:, cls_offset:]
        B, T, D = patch.shape
        tokens = (patch.reshape(B * T, D) - dim_mean).float()
        with torch.no_grad():
            _, _, x_hat = sae(tokens)
        reconstructed = (x_hat + dim_mean).reshape(B, T, D).to(hidden.dtype)
        if cls_offset:
            reconstructed = torch.cat([hidden[:, :cls_offset], reconstructed], dim=1)
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
