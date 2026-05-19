"""
Smoke test for MCG-NJU/videomae-base-finetuned-ssv2.

Checks:
  - Model loads and produces logits of shape (1, 174)
  - Forward hooks on encoder layers 5, 7, 9, 11 capture activations of shape (1, 1568, 768)
    (8 × 14 × 14 = 1568 patch tokens; no CLS token present in this model)
"""

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

MODEL_ID = "MCG-NJU/videomae-base-finetuned-ssv2"
HOOK_LAYERS = [5, 7, 9, 11]
EXPECTED_LOGITS_SHAPE = (1, 174)
EXPECTED_ACT_SHAPE = (1, 1568, 768)  # (batch, patches, hidden)


def main() -> bool:
    results: dict[str, dict] = {}

    print(f"Loading model: {MODEL_ID}")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
    model.eval()

    # 16 random uint8 frames, each 224×224×3
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(16)]

    # Register hooks — exclude CLS token (position 0) from captured activations
    activations: dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx in HOOK_LAYERS:
        def _make_hook(idx: int):
            def _hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                print(f"  layer {idx} raw shape: {tuple(hidden.shape)}")
                activations[idx] = hidden.detach()
            return _hook

        hooks.append(
            model.videomae.encoder.layer[layer_idx].register_forward_hook(_make_hook(layer_idx))
        )

    inputs = processor(frames, return_tensors="pt")

    print("\n--- Raw hook shapes (before CLS slice) ---")
    with torch.no_grad():
        outputs = model(**inputs)

    for h in hooks:
        h.remove()

    # --- check logits shape ---
    logits_shape = tuple(outputs.logits.shape)
    results["logits_shape"] = {
        "pass": logits_shape == EXPECTED_LOGITS_SHAPE,
        "expected": EXPECTED_LOGITS_SHAPE,
        "got": logits_shape,
    }

    # --- check activation shapes ---
    for layer_idx in HOOK_LAYERS:
        captured = activations.get(layer_idx)
        if captured is None:
            act_shape = None
        else:
            act_shape = tuple(captured.shape)
        results[f"layer_{layer_idx}_activations"] = {
            "pass": act_shape == EXPECTED_ACT_SHAPE,
            "expected": EXPECTED_ACT_SHAPE,
            "got": act_shape,
        }

    # --- summary ---
    print("\n=== Smoke Test Summary ===")
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result["pass"] else "FAIL"
        if not result["pass"]:
            all_pass = False
        if result["pass"]:
            print(f"  [{status}] {name}: {result['got']}")
        else:
            print(f"  [{status}] {name}: expected {result['expected']}, got {result['got']}")

    print(f"\nResult: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
