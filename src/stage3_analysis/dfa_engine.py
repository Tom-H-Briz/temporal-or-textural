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
    ["per_feature_summary", "correct_class_logit", "correct", "predicted_class"],
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
        self._sae.eval()
        for p in self._sae.parameters():
            p.requires_grad_(False)

        target_layer = self._model.videomae.encoder.layer[self._layer]
        self._hook_handle = target_layer.register_forward_hook(self._splice_hook)

        return self

    def _splice_hook(self, module, input, output):
        """
        Captures layer output, encodes to z (no_grad), detaches z, sets
        requires_grad_(True), decodes (tracked), adds mean back, returns
        reconstruction — splice is inside the hook.
        """
        hidden = output[0] if isinstance(output, tuple) else output  # (1, T, D)
        B, T, D = hidden.shape

        tokens = (hidden.reshape(B * T, D) - self._dim_mean).float()

        with torch.no_grad():
            _, z_raw = self._sae.encode(tokens)          # (T, dict_size)

        z = z_raw.detach().requires_grad_(True)           # leaf — accumulates grad
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

        self._z.grad = None
        logits[correct_class_idx].backward()

        grad_z = self._z.grad                              # (T, dict_size), signed
        dfa_tensor = grad_z * self._z.detach()             # (T, dict_size), signed
        per_feature_summary = (
            dfa_tensor.abs().sum(dim=0).detach().float()   # (dict_size,), float32
        )

        self._z.grad = None
        self._z = None

        return DFAResult(
            per_feature_summary=per_feature_summary,
            correct_class_logit=correct_class_logit_val,
            correct=correct,
            predicted_class=predicted_class,
        )

    def __exit__(self, *args) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        del self._model, self._sae, self._dim_mean
        self._model = self._sae = self._dim_mean = None
        torch.cuda.empty_cache()
