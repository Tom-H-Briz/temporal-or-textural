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
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from sae import BatchTopKSAE
from ToT_utils import MODEL_REGISTRY, gather_by_position

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
        "per_position_abs",     # (num_positions, dict_size) float32 or None
        "per_position_signed",  # (num_positions, dict_size) float32 or None
    ],
    defaults=(None, None),
)

# ---------------------------------------------------------------------------
# Model-specific preprocessors
# ---------------------------------------------------------------------------


def _preprocess_clip(
    clip: Path, num_frames: int, processor, device: str
) -> torch.Tensor:
    container = av.open(str(clip))
    frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n       = len(frames)
    indices = torch.linspace(0, n - 1, num_frames).long().tolist()
    sampled = [frames[i] for i in indices]
    return processor(sampled, return_tensors="pt")["pixel_values"].to(device)

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
        sae_k: int | None = None,
    ) -> None:
        self.model_flag = model_flag
        self.sae_path   = Path(sae_path)
        self.dim_mean_path = Path(dim_mean_path)
        self._layer  = layer
        self.device  = device
        self._sae_k  = sae_k   # fallback for old checkpoints without sae_k field

        self._model       = None
        self._sae         = None
        self._dim_mean    = None
        self._processor   = None
        self._hook_handle = None
        self._cls_offset: int            = 0
        self._num_frames:  int           = 16
        self._z:           torch.Tensor | None = None
        self._z_override:  torch.Tensor | None = None

    def __enter__(self) -> "DFAEngine":
        model_cfg        = MODEL_REGISTRY[self.model_flag]
        device           = self.device
        self._layer      = self._layer if self._layer is not None else 7
        self._cls_offset = model_cfg["cls_offset"]
        self._num_frames = model_cfg["num_frames"]

        self._processor = model_cfg["processor_class"].from_pretrained(model_cfg["checkpoint"])
        self._model     = model_cfg["model_class"].from_pretrained(model_cfg["checkpoint"])
        self._model.to(device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._dim_mean = torch.load(self.dim_mean_path, weights_only=True).to(device)

        ckpt       = torch.load(self.sae_path, weights_only=True, map_location=device)
        state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
        nb_concepts = state_dict["dictionary._weights"].shape[0]
        sae_k = ckpt.get("sae_k") or self._sae_k
        if sae_k is None:
            raise ValueError(f"sae_k not in checkpoint and not passed to constructor: {self.sae_path}")
        top_k = sae_k * model_cfg["num_patch_tokens"]

        self._sae = BatchTopKSAE(input_shape=model_cfg["hidden_dim"],
                                  nb_concepts=nb_concepts, top_k=top_k, device=device)
        self._sae.load_state_dict(state_dict)
        self._sae.train()
        dummy = torch.zeros(model_cfg["num_patch_tokens"], model_cfg["hidden_dim"], device=device)
        with torch.no_grad():
            self._sae.encode((dummy - self._dim_mean).float())
        for p in self._sae.parameters():
            p.requires_grad_(False)
        self._sae.eval()

        self._hook_handle = model_cfg["layer_getter"](self._model, self._layer) \
            .register_forward_hook(self._splice_hook)
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
        hidden     = output[0] if isinstance(output, tuple) else output
        cls_offset = self._cls_offset
        cls        = hidden[:, :cls_offset]   # (B, 1, D) for TF; empty slice for VideoMAE
        patches    = hidden[:, cls_offset:]   # (B, T, D)
        B, T, D    = patches.shape

        if self._z_override is not None:
            z = self._z_override
        else:
            tokens = (patches.reshape(B * T, D) - self._dim_mean).float()
            with torch.no_grad():
                _, z_raw = self._sae.encode(tokens)
            z = z_raw.detach().requires_grad_(True)
            self._z = z

        recon = self._sae.decode(z)
        recon = (recon + self._dim_mean).to(hidden.dtype).reshape(B, T, D)
        out   = torch.cat([cls, recon], dim=1) if cls_offset else recon
        return (out,) + output[1:] if isinstance(output, tuple) else out

    def run(self, clip: Path, correct_class_idx: int,
            return_per_position: bool = False) -> DFAResult:
        """Forward pass with SAE splice, then backward from correct-class logit."""
        pixel_values = _preprocess_clip(clip, self._num_frames, self._processor, self.device)
        return self.run_pixels(pixel_values, correct_class_idx,
                               return_per_position=return_per_position)

    def run_pixels(self, pixel_values: torch.Tensor, correct_class_idx: int,
                   return_per_position: bool = False) -> DFAResult:
        """Same as run() but accepts pre-computed pixel_values — use for in-memory perturbations."""
        self._z = None
        model_output = self._model(pixel_values=pixel_values)
        logits = model_output.logits.squeeze(0)            # (num_classes,)

        if self._z is None:
            raise RuntimeError(
                f"DFA hook did not fire — check layer={self._layer} for {self.model_flag}"
            )

        predicted_class = int(logits.argmax().item())
        correct = predicted_class == correct_class_idx
        correct_class_logit_val = float(logits[correct_class_idx].detach().item())
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

        # Optional per-position (VM: tubelet, TF: frame) aggregation, OFF BY DEFAULT
        per_position_abs = None
        per_position_signed = None
        if return_per_position:
            grouped = gather_by_position(dfa_tensor, self.model_flag)  # (num_positions, N_SPATIAL, dict_size)
            per_position_abs    = grouped.abs().sum(dim=1).detach().float().cpu()
            per_position_signed = grouped.sum(dim=1).detach().float().cpu()

        per_feature_summary = (
            dfa_tensor.abs().sum(dim=0).detach().float().cpu()   # (dict_size,), float32
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
            per_position_abs=per_position_abs,
            per_position_signed=per_position_signed,
        )

    def run_ablated(
        self,
        pixel_values: torch.Tensor,
        correct_class_idx: int,
        ablate_indices: list[int],
        z_cache: torch.Tensor,
    ) -> tuple[float, int, bool, np.ndarray]:
        """Forward-only ablation pass. Zeros ablate_indices in z before decode.
        ablate_indices=[] gives the unablated baseline. No backward pass."""
        z = z_cache.clone()
        if ablate_indices:
            z[:, ablate_indices] = 0.0
        self._z_override = z
        try:
            with torch.no_grad():
                output = self._model(pixel_values=pixel_values)
        finally:
            self._z_override = None
        logits = output.logits.squeeze(0)
        predicted = int(logits.argmax().item())
        all_logits = logits.detach().cpu().float().numpy()
        return float(logits[correct_class_idx].item()), predicted, predicted == correct_class_idx, all_logits

    def get_z_pixels(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Like get_z() but accepts pre-computed pixel_values — use for in-memory perturbations."""
        self._z = None
        with torch.no_grad():
            self._model(pixel_values=pixel_values)
        if self._z is None:
            raise RuntimeError(f"DFA hook did not fire — check layer={self._layer}")
        z = self._z.detach().clone()
        self._z = None
        return z

    def get_z(self, clip: Path) -> torch.Tensor:
        """Forward pass in no_grad — returns z (num_tokens, dict_size) without backward."""
        pixel_values = _preprocess_clip(clip, self._num_frames, self._processor, self.device)
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
