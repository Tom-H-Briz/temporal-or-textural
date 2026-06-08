"""
Direct Feature Attribution (DFA) engine.

Follows Marks et al. (arXiv:2403.19647): BatchTopKSAE spliced into the model
forward pass at a target layer; the correct-class logit is backpropagated
through the SAE to obtain per-feature gradient × activation scores.

Usage:
    with DFAEngine("videomae", sae_path, dim_mean_path, layer=7, device="cuda") as engine:
        for clip_path, class_idx in clip_iterator:
            result = engine.run(clip_path, class_idx)
"""

import collections
import math
import sys
from pathlib import Path

import av
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from sae import BatchTopKSAE

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

DFAResult = collections.namedtuple(
    "DFAResult",
    [
        "per_feature_summary",
        "correct_class_logit",
        "correct",
        "predicted_class",
        "entropy_normalised",
        "logit_margin",
        "all_logits",
        "signed_feature_summary",
        "token_fire_counts",
    ],
)

# ---------------------------------------------------------------------------
# Model-specific preprocessors
# ---------------------------------------------------------------------------


def _preprocess_videomae(
    clip: Path, num_frames: int, processor, device: str
) -> torch.Tensor:
    container = av.open(str(clip))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()

    n = len(frames)
    indices = torch.linspace(0, n - 1, num_frames).long().tolist()
    sampled = [frames[i] for i in indices]

    return processor(sampled, return_tensors="pt")["pixel_values"].to(device)


# ---------------------------------------------------------------------------
# Model config block — add entries here to support new architectures
# ---------------------------------------------------------------------------

MODEL_CONFIGS: dict[str, dict] = {
    "videomae": {
        "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
        "num_frames": 16,
        "hidden_dim": 768,
        "num_tokens": 1568,
        "hook_layer": 7,
        "nb_concepts": 6144,
        "sae_k": 128,
        "preprocessor": _preprocess_videomae,
    },
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def compute_entropy_normalised(logits: torch.Tensor) -> float:
    """
    Shannon entropy of the softmax distribution, normalised to [0, 1].

    H = -Σ p_i * log(p_i),  H_normalised = H / log(num_classes)

    0 * log(0) = 0 by convention (clamped before log).
    Returns a Python float.
    """
    with torch.no_grad():
        probs = torch.softmax(logits.detach(), dim=0)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
        return entropy / math.log(probs.shape[0])


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class DFAEngine:
    """
    Context manager: loads model + SAE on enter, cleans up GPU memory on exit.
    Each run() call does one forward/backward pass for a single clip.
    """

    def __init__(
        self,
        model_flag: str,
        sae_path: str | Path,
        dim_mean_path: str | Path,
        layer: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_flag = model_flag
        self.sae_path = Path(sae_path)
        self.dim_mean_path = Path(dim_mean_path)
        self._layer = layer  # None → resolved from MODEL_CONFIGS in __enter__
        self.device = device

        self._model = None
        self._sae = None
        self._dim_mean = None
        self._processor = None
        self._hook_handle = None
        self._z: torch.Tensor | None = None
        self._z_override: torch.Tensor | None = None  # set externally for ablation passes

    def __enter__(self) -> "DFAEngine":
        cfg = MODEL_CONFIGS[self.model_flag]
        device = self.device
        self._layer = self._layer if self._layer is not None else cfg["hook_layer"]

        self._processor = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
        self._model = VideoMAEForVideoClassification.from_pretrained(cfg["model_id"])
        self._model.to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._dim_mean = torch.load(self.dim_mean_path, weights_only=True).to(device)

        self._sae = BatchTopKSAE(
            input_shape=cfg["hidden_dim"],
            nb_concepts=cfg["nb_concepts"],
            top_k=cfg["sae_k"] * cfg["num_tokens"],
            device=device,
        )
        self._sae.load_state_dict(
            torch.load(self.sae_path, weights_only=True, map_location=device)
        )
        # running_threshold is not persisted in the checkpoint — initialise with a
        # dummy train-mode pass so eval() asserts do not fire (same approach as
        # spliced_accuracy.py)
        self._sae.train()
        dummy = torch.zeros(cfg["num_tokens"], cfg["hidden_dim"], device=device)
        with torch.no_grad():
            self._sae.encode((dummy - self._dim_mean).float())
        # Freeze before eval() so _fused_dictionary is computed with requires_grad=False
        # weights, giving it no grad_fn. Without this, backward() frees _fused_dictionary's
        # saved tensors on the first clip and the second clip raises "backward through graph
        # a second time".
        for p in self._sae.parameters():
            p.requires_grad_(False)
        self._sae.eval()

        target_layer = self._model.videomae.encoder.layer[self._layer]
        self._hook_handle = target_layer.register_forward_hook(self._splice_hook)

        return self

    def _splice_hook(self, module, input, output):
        """
        Captures layer output, encodes to z (no_grad), detaches z, sets
        requires_grad_(True), decodes (tracked), adds mean back, returns
        reconstruction — splice is inside the hook.

        If _z_override is set the encode step is skipped and the provided z is
        decoded directly (used by ablation passes in no_grad context).
        """
        assert self._sae is not None and self._dim_mean is not None
        hidden = output[0] if isinstance(output, tuple) else output  # (1, T, D)
        B, T, D = hidden.shape

        if self._z_override is not None:
            z = self._z_override                          # (T, dict_size), caller-provided
        else:
            tokens = (hidden.reshape(B * T, D) - self._dim_mean).float()
            with torch.no_grad():
                _, z_raw = self._sae.encode(tokens)       # (T, dict_size)
            z = z_raw.detach().requires_grad_(True)        # leaf — accumulates grad
            self._z = z

        recon = self._sae.decode(z)                       # (T, hidden_dim), tracked through z
        recon = (recon + self._dim_mean).to(hidden.dtype).reshape(B, T, D)

        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon

    def run(self, clip: Path, correct_class_idx: int) -> DFAResult:
        """Forward pass with SAE splice, then backward from correct-class logit."""
        cfg = MODEL_CONFIGS[self.model_flag]
        pixel_values = cfg["preprocessor"](
            clip, cfg["num_frames"], self._processor, self.device
        )

        self._z = None
        model_output = self._model(pixel_values=pixel_values)
        logits = model_output.logits.squeeze(0)            # (num_classes,)

        if self._z is None:
            raise RuntimeError(
                f"DFA hook did not fire — check layer={self._layer} for {self.model_flag}"
            )

        predicted_class = int(logits.argmax().item())
        correct = predicted_class == correct_class_idx
        correct_class_logit_val = float(logits[correct_class_idx].item())
        entropy_norm = compute_entropy_normalised(logits)

        all_logits = logits.detach().cpu().float()
        top2_vals = all_logits.topk(2).values
        second_highest = top2_vals[1].item()
        logit_margin = correct_class_logit_val - second_highest

        self._z.grad = None
        logits[correct_class_idx].backward()

        grad_z = self._z.grad                              # (T, dict_size), signed
        z_detached = self._z.detach()
        dfa_tensor = grad_z * z_detached                   # (T, dict_size), signed
        per_feature_summary = (
            dfa_tensor.abs().sum(dim=0).detach().float()   # (dict_size,), float32
        )
        signed_feature_summary = dfa_tensor.sum(dim=0).detach().float().cpu()
        token_fire_counts = (z_detached > 0).sum(dim=0).cpu().to(torch.int32)

        self._z.grad = None
        self._z = None

        return DFAResult(
            per_feature_summary=per_feature_summary,
            correct_class_logit=correct_class_logit_val,
            correct=correct,
            predicted_class=predicted_class,
            entropy_normalised=entropy_norm,
            logit_margin=logit_margin,
            all_logits=all_logits,
            signed_feature_summary=signed_feature_summary,
            token_fire_counts=token_fire_counts,
        )

    def get_z(self, clip: Path) -> torch.Tensor:
        """Forward pass in no_grad — returns z (num_tokens, dict_size) without backward."""
        cfg = MODEL_CONFIGS[self.model_flag]
        pixel_values = cfg["preprocessor"](
            clip, cfg["num_frames"], self._processor, self.device
        )
        self._z = None
        with torch.no_grad():
            self._model(pixel_values=pixel_values)
        if self._z is None:
            raise RuntimeError(
                f"DFA hook did not fire — check layer={self._layer} for {self.model_flag}"
            )
        z = self._z.detach().clone()
        self._z = None
        return z

    def __exit__(self, *args) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        del self._model, self._sae, self._dim_mean
        self._model = self._sae = self._dim_mean = None
        torch.cuda.empty_cache()
