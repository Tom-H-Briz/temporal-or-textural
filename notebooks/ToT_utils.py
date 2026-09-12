"""
Shared utilities for temporal-or-textural notebooks.
"""

import json
import zlib
from functools import partial
from pathlib import Path

import av
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    TimesformerForVideoClassification,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    VivitForVideoClassification,
    VivitImageProcessor,
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
    "vivit": {
        "model_class":      VivitForVideoClassification,
        "num_frames":       32,
        "processor_class":  VivitImageProcessor,
        "cls_offset":       1,
        "layer_getter":     lambda model, i: model.vivit.encoder.layer[i],
        "hidden_dim":       768,
        "num_patch_tokens": 3136,
        "position_label":   "tubelet",
        "frame_sample_rate": 2,  # paper §4.1: 32 frames at stride 2 (VM's kinetics protocol is 4)
    },
}

# (model_name, dataset_name) -> HF checkpoint string. The only place a finetuned
# checkpoint string lives — checkpoint identity is a (backbone x dataset) product.
CHECKPOINT_REGISTRY: dict[tuple[str, str], str] = {
    ("videomae", "ssv2"):         "MCG-NJU/videomae-base-finetuned-ssv2",
    ("timesformer", "ssv2"):      "facebook/timesformer-base-finetuned-ssv2",
    ("videomae", "kinetics400"):  "MCG-NJU/videomae-base-finetuned-kinetics",
    ("vivit", "kinetics400"):     "google/vivit-b-16x2-kinetics400",
}

# k -> expansion. This project has only ever trained these two SAE configs — not a
# general rule, just the two points this pipeline has data for.
_SAE_EXPANSION_FOR_K = {64: 8, 128: 16}


def resolve_sae_checkpoint(
    model_flag: str,
    layer: int,
    dataset_name: str = "ssv2",
    sae_k: int = 64,
    job_label: str = "7ep",
) -> dict:
    """Locate an SAE checkpoint + its dim_mean under the current (post-30/07
    bias-fix) naming scheme, and read nb_concepts/sae_k from the checkpoint's own
    weights — never hardcoded, since k64/x8 and k128/x16 share no fixed size.

    Was duplicated near-identically across dfa_per_tubelet_mass.py,
    z_position_lock_extraction.py, dfa_mass_delta_vm.py, and run_ablation.py, all
    still pointed at the pre-fix filename scheme (now only in outputs/sae/legacy/)
    — same precedent as gather_by_position: one shared function beats N copies
    that can go stale in lockstep.
    """
    sae_dir = ROOT / "outputs" / "sae"
    if model_flag == "videomae":
        expansion = _SAE_EXPANSION_FOR_K[sae_k]
        sae_path = sae_dir / f"sae_vmae_{dataset_name}_k{sae_k}_x{expansion}_l{layer}_job{job_label}_best.pt"
        dim_mean = sae_dir / f"vmae_{dataset_name}_layer{layer}_dim_mean.pt"
    elif model_flag == "timesformer":
        assert dataset_name == "ssv2", "TimeSformer has no non-SSv2 checkpoints in this project"
        job_label = str(layer)  # TF's own established convention — layer is the job label
        matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected 1 TF checkpoint for layer {layer}, found: {matches}")
        sae_path = matches[0]
        dim_mean = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    elif model_flag == "vivit":
        assert dataset_name == "kinetics400", "ViViT has no non-Kinetics-400 checkpoints in this project"
        expansion = _SAE_EXPANSION_FOR_K[sae_k]
        sae_path = sae_dir / f"sae_vivit_{dataset_name}_k{sae_k}_x{expansion}_l{layer}_job{job_label}_best.pt"
        dim_mean = sae_dir / f"vivit_{dataset_name}_layer{layer}_dim_mean.pt"
    else:
        raise ValueError(f"Unknown model_flag: {model_flag!r}")

    if not sae_path.exists():
        raise FileNotFoundError(f"{model_flag} SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")

    ckpt = torch.load(sae_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["sae_state_dict"] if isinstance(ckpt, dict) and "sae_state_dict" in ckpt else ckpt
    nb_concepts = state_dict["dictionary._weights"].shape[0]
    ckpt_sae_k  = ckpt.get("sae_k") if isinstance(ckpt, dict) else None
    return {
        "sae_path": str(sae_path), "dim_mean_path": str(dim_mean),
        "sae_k": ckpt_sae_k or sae_k, "nb_concepts": nb_concepts, "job_label": job_label,
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
    elif model_flag == "vivit":
        # conv3d tubelet embedding flattens (T, H, W) time-major — same layout as VM
        return tokens.reshape(num_positions, N_SPATIAL, *tokens.shape[1:])
    else:
        raise ValueError(f"No position-gather rule registered for model_flag={model_flag!r}")


def _strip_brackets(template: str) -> str:
    return template.replace("[", "").replace("]", "")


def _deterministic_seed(clip_id: str) -> int:
    """RNG seed for shuffle conditions. SSv2 clip_ids are numeric strings — int()
    used directly, preserving the exact seed already validated against real SSv2
    output. K400 clip_ids are YouTube-style strings (e.g. 'XAoQRtv6OyA_000088_000098')
    that int() can't parse — every K400 clip was hitting this and getting skipped
    (confirmed 31/07 on Isambard). Falls back to a deterministic hash only when
    int() fails, so SSv2 is untouched. Moved here from position_lock_extraction.py
    (03/08) so dfa_mass_delta_vm.py's K400 path can reuse it instead of re-hitting
    the same bug."""
    try:
        return int(clip_id) % 2**32
    except ValueError:
        return zlib.crc32(clip_id.encode()) % 2**32


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


def resolve_k400_label2id(model_flag: str) -> dict[str, int]:
    """label2id for a Kinetics-400 checkpoint. Most finetuned checkpoints carry real
    class names in their config, but google/vivit-b-16x2-kinetics400 ships generic
    LABEL_N placeholders (id2label is LABEL_0..LABEL_399) — resolving from the
    config there yields a map where every real class name lookup misses. Fall back
    to the canonical alphabetical 400-class list (notebooks/k400_label_names.json,
    extracted from the VM-kinetics checkpoint config — verified alphabetical), which
    is the ordering both conversions follow. Correctness is gated downstream by the
    known-clip smoke test / baseline accuracy (right map ~60%+ top-1, wrong ~0.25%).
    """
    checkpoint = CHECKPOINT_REGISTRY[(model_flag, "kinetics400")]
    label2id = AutoConfig.from_pretrained(checkpoint).label2id
    if not any(k.startswith("LABEL_") for k in label2id):
        return label2id
    names = json.loads((Path(__file__).parent / "k400_label_names.json").read_text())
    return {name: i for i, name in enumerate(names)}


def load_clips_kinetics(
    manifest_path: str, video_dir: str, model_flag: str
) -> list[tuple[str, int, Path]]:
    """K400's equivalent of load_metadata() above: given a static SL-subset
    manifest, return every (clip_id, class_id, video_path) triple ready to feed
    a DFAEngine run. Shared by position_lock_extraction.py, dfa_mass_delta_vm.py,
    and ablation_cross_l5_l7.py (03/08) — was duplicated near-identically across
    the first two, same "one shared function, not N copies" precedent as
    resolve_sae_checkpoint above.

    manifest_path: outputs/Laura_SL/k400_manifest_SL_subset.json, built once by
    notebooks/build_k400_sl_manifest.py. Schema: {"temporal": [...], "static": [...]},
    each entry {"id": <clip filename stem>, "label": <K400 class name>}. This is
    the ENTIRE clip population — no held-out/train split, no correctness gate,
    same scope as SSv2's manifest_SL_subset.json (see 03/08 CC brief: SSv2 has
    never excluded SAE-training clips from analysis, so K400 matches that rather
    than diverging from it).

    label2id is resolved here, at call time, rather than baked into the manifest,
    because it's checkpoint-specific (the finetuned model's own output-index
    space) — not part of "which clips are in the SL population", which is fixed.
    Keeping the two separate means a checkpoint swap can't silently change which
    clips are in scope, only how their labels map to logit indices.

    Clips are filtered to ones whose video file actually exists on disk — the
    manifest is built from val.csv metadata alone, with no guarantee the .mp4 is
    present locally (K400 clips are Isambard-synced separately, per this
    project's data-sync convention).
    """
    checkpoint = CHECKPOINT_REGISTRY[(model_flag, "kinetics400")]
    label2id   = resolve_k400_label2id(model_flag)
    with open(manifest_path) as f:
        manifest = json.load(f)

    video_dir_path = Path(video_dir)
    result = []
    for entries in manifest.values():
        for entry in entries:
            cid  = label2id.get(entry["label"])
            path = video_dir_path / f"{entry['id']}.mp4"
            if cid is not None and path.exists():
                result.append((entry["id"], cid, path))
    return result


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


def sample_frames_ssv2(n: int, num_frames: int) -> list[int]:
    """VideoMAE's SSv2 finetuning protocol: num_frames spread evenly across the
    whole clip (uniform sampling) — matches SSv2's short clips closely."""
    return torch.linspace(0, n - 1, num_frames).long().tolist()


def sample_frames_kinetics(n: int, num_frames: int, frame_sample_rate: int = 4) -> list[int]:
    """VideoMAE's Kinetics-400 finetuning/eval protocol: dense sampling — a single,
    deterministic num_frames*frame_sample_rate-frame window (center-positioned),
    taken from the clip rather than stretched across it. Default 16*4=64 frames
    (~2.1-2.56s at typical fps). Falls back to sample_frames_ssv2 if the clip is
    shorter than the window — same clamping behavior as the reference implementation.
    """
    window = num_frames * frame_sample_rate
    if window >= n:
        return sample_frames_ssv2(n, num_frames)
    start = (n - window) // 2
    return torch.linspace(start, start + window - 1, num_frames).long().tolist()


# dataset_name -> frame_sampler. Single lookup shared by every caller that builds
# an SSv2ClipDataset for a specific dataset, so the sampler choice can't drift
# between train_sae.py / spliced_accuracy_vm.py / profile_activations.py.
FRAME_SAMPLERS = {
    "ssv2":         sample_frames_ssv2,
    "kinetics400":  sample_frames_kinetics,
}


def get_frame_sampler(dataset_name: str, model_cfg: dict):
    """Dataset sampler with the model's own frame_sample_rate applied. The rate is
    a backbone property (VM's kinetics protocol: 4; ViViT: 2 per paper §4.1), so it
    lives in MODEL_REGISTRY and is partial-applied here — one shared construction
    point so validation, SAE training and DFA stages can't drift onto different
    strides. Datasets whose sampler takes no rate (ssv2) get the plain sampler.
    """
    sampler = FRAME_SAMPLERS[dataset_name]
    if dataset_name == "kinetics400":
        return partial(sampler, frame_sample_rate=model_cfg.get("frame_sample_rate", 4))
    return sampler


class SSv2ClipDataset(Dataset):
    """
    Loads video clips from explicit paths.

    With labels: __getitem__ returns (pixel_values, label) — for classification.
    Without labels: __getitem__ returns pixel_values — for SAE training.

    frame_sampler: (n_decoded_frames, num_frames) -> frame indices to keep. Defaults
    to sample_frames_ssv2 (uniform, correct for SSv2/TimeSformer callers that predate
    the dataset axis and never pass one). Kinetics callers must pass
    sample_frames_kinetics explicitly — getting this wrong is silent, not an error.
    """

    def __init__(
        self,
        clip_paths: list[Path],
        processor,
        num_frames: int,
        labels: list[int] | None = None,
        frame_sampler=sample_frames_ssv2,
    ) -> None:
        assert labels is None or len(labels) == len(clip_paths)
        self.clip_paths = clip_paths
        self.processor = processor
        self.num_frames = num_frames
        self.labels = labels
        self.frame_sampler = frame_sampler

    def __len__(self) -> int:
        return len(self.clip_paths)

    def __getitem__(self, idx: int):
        # Kinetics-400's YouTube-sourced download reliably contains some corrupted/
        # empty clips at ~20k scale; SSv2's curated webm set doesn't hit this, but
        # a single bad file must not crash an entire epoch — retry with the next
        # clip (bounded) rather than propagate the decode error.
        frames = None
        for _ in range(5):
            try:
                container = av.open(str(self.clip_paths[idx]))
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
                container.close()
                break
            except Exception as e:
                print(f"  Warning: unreadable clip {self.clip_paths[idx]} ({e}); trying next")
                idx = (idx + 1) % len(self.clip_paths)
        if frames is None:
            raise RuntimeError(f"5 consecutive unreadable clips starting near idx {idx}")

        n = len(frames)
        indices = self.frame_sampler(n, self.num_frames)
        sampled = [frames[i] for i in indices]

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)

        if self.labels is not None:
            return pixel_values, self.labels[idx]
        return pixel_values
