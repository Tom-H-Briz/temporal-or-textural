# DFA Engine — Specification

**Project:** Temporal or Textural? Mechanistic Interpretability of Video Transformers Using Sparse Autoencoders  
**Date:** 04 June 2026  
**Status:** Ready for implementation  
**File:** `src/stage3_analysis/dfa_engine.py`

---

## Overview

A reusable, model-agnostic engine for computing Direct Feature Attribution (DFA) scores for a single clip. The engine takes a raw clip and a correct class index, handles all preprocessing, forward pass, SAE splice, and backward pass internally, and returns a per-feature summary vector.

The engine is designed to be extended to new models by adding a model config entry at the top of the file. No other changes required.

Follows Marks et al. (arXiv:2403.19647): SAE spliced into forward pass, correct-class logit backpropagated through SAE to obtain gradient × activation scores.

---

## Model Config Block

At the top of `dfa_engine.py`, a dict keyed by model flag defines all model-specific parameters:

```python
MODEL_CONFIGS = {
    "videomae": {
        "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
        "num_frames": 16,
        "hidden_dim": 768,
        "num_tokens": 1568,        # CLS excluded
        "hook_layer": 7,           # overridden by engine config if specified
        "nb_concepts": 6144,       # SAE dictionary size (8x expansion)
        "sae_k": 128,              # SAE sparsity parameter
        "preprocessor": _preprocess_videomae,  # function reference
    },
    # future models added here — no other changes required
}
```

SAE architecture parameters (`nb_concepts`, `sae_k`) live in `MODEL_CONFIGS` alongside model-specific fields. The engine is fully self-describing per model flag — no magic numbers anywhere in the computation path. Adding a new model with a different SAE configuration requires only a new dict entry.

The preprocessor function is defined in the same file, takes a raw clip, returns a model-ready tensor. Swapping model flag at instantiation is the only change needed to run the engine on a different architecture.

---

## Class Interface

### `DFAEngine(model_flag, sae_path, dim_mean_path, layer, device)`

**Parameters:**
- `model_flag` — string key into `MODEL_CONFIGS`, e.g. `"videomae"`
- `sae_path` — path to SAE checkpoint, e.g. `"outputs/sae/sae_layer_7.pt"`
- `dim_mean_path` — path to per-dim mean vector, e.g. `"outputs/sae/layer7_dim_mean.pt"`
- `layer` — integer, target layer index. Overrides model config default if provided.
- `device` — `"cuda"` or `"cpu"`

### `__init__(self, ...)`
Stores config. No model loading. No device allocation.

### `__enter__(self)`
- Loads model per `model_flag` from `MODEL_CONFIGS`
- Loads SAE from `sae_path`
- Loads per-dim mean vector from `dim_mean_path`
- Moves all to `device`
- Runs a dummy zero forward pass through the SAE in train mode to initialise the running threshold `θ` — threshold is not persisted in the checkpoint so must be initialised before eval. Dummy pass sets `θ` to ~0.
- Calls `sae.eval()` explicitly — locks `θ` for consistent inference and ensures `DictionaryLayer.get_dictionary()` has `_fused_dictionary` set (required by `overcomplete` library)
- Sets model to `eval()` mode
- Registers forward hook on target layer
- Returns `self`

### `__exit__(self, *args)`
- Removes forward hook
- Deletes model, SAE, dim_mean references
- Calls `torch.cuda.empty_cache()`
- Ensures GPU memory is released even if analysis job raises an exception

### `run(self, clip, correct_class_idx)`
Single clip forward/backward pass. See mechanics below.

**Parameters:**
- `clip` — raw clip, any format. Engine applies model-specific preprocessing internally.
- `correct_class_idx` — integer. Caller's responsibility.

**Returns:** named tuple or dict:
- `per_feature_summary` — `(dict_size,)` float32, absolute value summed across tokens
- `correct_class_logit` — scalar float
- `correct` — boolean, whether top-1 prediction matches `correct_class_idx`
- `predicted_class` — integer, top-1 predicted class index

---

## `run()` Mechanics

**Preprocessing:**
```
clip → model-specific preprocessor → model-ready tensor
```

**Forward pass:**
1. Pass preprocessed clip through frozen model — hook fires at layer 7
2. Hook captures layer 7 output `(num_tokens, hidden_dim)`
3. Hook subtracts per-dim mean — broadcast across tokens
4. Hook encodes under `torch.no_grad()` → `z` shape `(num_tokens, dict_size)`
5. Hook calls `z.detach().requires_grad_(True)` — `z` becomes a gradient leaf
6. Hook decodes outside `no_grad()` → reconstruction `(num_tokens, hidden_dim)`
7. Hook adds per-dim mean back to reconstruction
8. Hook returns reconstruction — layers 8+ operate on spliced activations
9. Model continues forward pass to classification head → logits `(num_classes,)`

**Backward pass:**
10. Extract `correct_class_logit = logits[correct_class_idx]`
11. Zero any existing gradients on `z`
12. `correct_class_logit.backward()`
13. `grad_z = z.grad` — shape `(num_tokens, dict_size)`, signed
14. `dfa_tensor = grad_z * z` — element-wise, signed, shape `(num_tokens, dict_size)`
15. `per_feature_summary = dfa_tensor.abs().sum(dim=0)` — shape `(dict_size,)`, absolute value summed across tokens

**Signed scores preserved in full:** positive DFA means feature increases correct-class logit; negative means it decreases it. Absolute value taken only for the per-feature summary, which is used for cumulative mass diagnostic. The sign information is available from the full `dfa_tensor` if needed for later analysis.

**Correct flag:**
16. `predicted_class = logits.argmax().item()`
17. `correct = (predicted_class == correct_class_idx)`

**Cleanup:**
18. Zero gradients on `z` after extraction
19. Detach tensors before returning

---

## `_preprocess_videomae(clip)` (private)

Model-specific preprocessor for VideoMAE. Input `clip` is a `Path` to a `.webm` file. Handles:
- Reading and decoding frames from WebM
- Frame sampling to `num_frames` (16)
- Normalisation per VideoMAE training statistics
- Tensor formatting to `(1, num_frames, channels, height, width)`
- Move to device

Future models add their own preprocessor function and reference it in `MODEL_CONFIGS`.

---

## Hook Management

Forward hook `_splice_hook` registered in `__enter__` on the target layer's output. The hook both captures and modifies the forward pass — it must return the SAE reconstruction so that layers 8+ operate on the spliced representation rather than the original layer 7 output.

Hook mechanics:
1. Captures layer 7 output `(num_tokens, hidden_dim)`
2. Encodes to `z` under `torch.no_grad()`
3. Calls `z.detach().requires_grad_(True)` — detach breaks gradient path into encoder (not needed); `requires_grad_(True)` makes `z` a leaf tensor that accumulates gradients from the backward pass through decoder and classification head
4. Decodes outside `no_grad()` context so gradient flows through `z`
5. Adds per-dim mean back to reconstruction
6. Returns reconstruction — model continues forward pass from layer 8 with spliced activations

Hook handle stored as instance attribute. Removed in `__exit__` via `handle.remove()`.

---

## Gradient Isolation

All model parameters and SAE weights have `requires_grad=False`. Only `z` (set via `detach().requires_grad_(True)` inside the hook) participates in the gradient computation. Autograd traces exactly the path:

```
correct_class_logit → model tail (layers 8–12) → spliced reconstruction → sae.decode(z) → z
```

`z.grad` is cleanly populated after `correct_class_logit.backward()`. No gradients flow into VideoMAE weights or SAE weights.

---

## Output

Named tuple `DFAResult`:

| Field | Type | Description |
|-------|------|-------------|
| `per_feature_summary` | `(dict_size,)` float32 | Absolute DFA summed across tokens |
| `correct_class_logit` | float | Correct-class logit scalar |
| `correct` | bool | Top-1 prediction matches correct class |
| `predicted_class` | int | Top-1 predicted class index |

---

## Usage Pattern

```python
with DFAEngine(
    model_flag="videomae",
    sae_path="outputs/sae/sae_layer_7.pt",
    dim_mean_path="outputs/sae/layer7_dim_mean.pt",
    layer=7,
    device="cuda"
) as engine:
    for clip, correct_class_idx in clip_iterator:
        result = engine.run(clip, correct_class_idx)
        # result.per_feature_summary, result.correct, etc.
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Context manager | `__enter__` / `__exit__` | Guaranteed lifecycle management; GPU memory released even on exception |
| Model flag | String key into `MODEL_CONFIGS` | Single change to swap model; all model-specific logic encapsulated |
| Preprocessing | Internal per model flag | Caller passes raw clip; engine handles format — analysis jobs are model-agnostic |
| Correct class index | Caller-provided | Engine has no knowledge of metadata or labels |
| Signed DFA | Preserved in `dfa_tensor` | Distinguishes features helping vs hurting classification |
| Per-feature summary | Absolute value, sum across tokens | Captures total causal influence for cumulative mass diagnostic |
| No file I/O | None in engine | Engine is a pure computation unit; analysis jobs handle storage |
| Batch size | 1 clip | Backward pass per clip; clean gradient isolation without complexity |
| Hook removal | In `__exit__` | Prevents accumulation; guaranteed removal even on exception |
| `sae.eval()` explicit | Yes | `overcomplete` library requires explicit eval call for `get_dictionary()` |
| `torch.cuda.empty_cache()` | In `__exit__` | Ensures GPU memory released for next context or subsequent jobs |

---

## Dependencies

- `torch` — forward/backward pass, gradient computation
- `transformers` — model loading per model flag
- `overcomplete` — SAE loading (`KempnerInstitute/overcomplete`)
- `ToT_utils` — `MODEL_ID`, `NUM_FRAMES`, `NUM_CLASSES` (referenced via `MODEL_CONFIGS`)
- `pathlib` — path handling
- `collections.namedtuple` — `DFAResult`

---

## Explicit Non-Goals

- No file I/O of any kind
- No metadata handling
- No iteration over clip lists — single clip only
- No threshold decisions
- No TFR computation
- No batching across clips
- No weight updates
