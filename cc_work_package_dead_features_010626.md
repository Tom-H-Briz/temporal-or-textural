# Work Package: Dead Feature Diagnostic Sweep
**Date:** 01 June 2026  
**For:** Claude Code  
**Project:** Temporal or Textural? — `src/stage2_sae/train_sae.py`  
**Status:** Ready to implement

---

## Context

Initial SAE training on VideoMAE layer 7 produces ~40% dead features increasing linearly through training. This is not acceptable — effective dictionary is ~3,700 not 6,144 features.

**Diagnosis:** BatchTopK at clip level (pool = 1568 tokens) allows the positional scaffold in VideoMAE's layer 7 residual stream to monopolise the shared budget. One dimension has mean=66.87 (~30σ above zero) and is essentially constant across all tokens and clips — any SAE feature dedicated to it fires on every token permanently, consuming budget. Per-dim mean subtraction (Gao et al. standard practice) removes this before the SAE sees the activations.

**What is working correctly:** L0 ~64 (sparsity correct), aux loss plumbed in correctly, R²=0.852.

**What is not working:** Aux loss α=0.03 is insufficient to overcome positional scaffold monopolisation. Top-50% dead feature rescue mask in overcomplete ignores the bottom 50% of dead features.

---

## Existing Code to Understand First

Before making any changes, read and understand:

- `src/stage2_sae/train_sae.py` — the existing training loop
- The `overcomplete` library's `BatchTopKSAE` class — specifically:
  - How the aux loss is computed and what α controls
  - The dead feature rescue function that uses the top-50% mask
  - Whether there is a zero-mask alternative (rescues all dead features, not just top 50%)
  - The `DictionaryLayer.get_dictionary()` quirk — requires `.eval()` to be called first to set `_fused_dictionary`

---

## Step 0 — Per-Dim Mean Subtraction (applies to ALL jobs)

This is a preprocessing change that goes into the training loop before anything else. All jobs in the sweep include this change.

**What to do:**

1. Before the training loop begins, compute the per-dimension mean of layer 7 activations across the training set. Shape: `(768,)`, dtype float32. Save to `outputs/sae/layer7_dim_mean.pt`.

2. At the hookpoint, subtract this mean from every token's activation before passing to the SAE:
```python
activations = activations - dim_mean  # dim_mean shape (768,) broadcasts correctly
```

3. The same subtraction must be applied at inference time (DFA analysis, spliced accuracy test) — the SAE was trained on mean-subtracted activations so inference must match.

**Activation statistics for reference (2k clip sample):**
- Global mean: 0.1825
- Per-dim mean max: 66.87 (the outlier dimension)
- Per-dim mean min: -7.26
- Per-dim std mean: 2.26

The 66.87-mean dimension is the primary target. After subtraction it becomes zero-mean with the same variance.

---

## The 8 Parallel Jobs

All jobs include Step 0 (per-dim mean subtraction). Each job runs for **1 epoch only** on Isambard. One epoch is sufficient to observe the dead feature trajectory.

| Job | SAE Type | k | α | Dead Feature Mask | Notes |
|-----|----------|---|---|-------------------|-------|
| A | BatchTopK | 64 | 0.03 | Top-50% | New baseline — mean subtraction only, everything else unchanged |
| B | BatchTopK | 64 | 0.1 | Top-50% | α sweep |
| C | BatchTopK | 64 | 0.3 | Top-50% | α sweep |
| D | BatchTopK | 64 | 1.0 | Top-50% | α sweep |
| E | BatchTopK | 64 | 0.03 | Full zero-mask | Full dead feature rescue — all dead features not just top 50% |
| F | BatchTopK | 96 | 0.03 | Top-50% | More budget per feature — may slow death rate even if root cause not fixed |
| G | BatchTopK | 128 | 0.03 | Top-50% | More budget per feature — larger test |
| H | BatchTopK | 64 | 0.03 | Top-50% | Diagnostic baseline — controls for per dim mean subtraction to normalise |

**For Job H (standard TopK):** `top_k = 64` per token, not `64 × 1568`. Standard TopK enforces exactly k=64 features per token independently. This is the comparison to confirm whether BatchTopK is the root cause.

**For Jobs F and G (increased k):** `top_k = k × 1568` passed to `BatchTopKSAE` as before — just with k=96 or k=128.

---

## What to Record Per Job

For each job, record at end of epoch 1:

```
Job X results:
- Dead features: N / 6144 (N%)
- Dead feature trajectory: linear / slowing / flat (qualitative)
- L0: (should be ~k)
- R²: 
- MSE:
- Notes:
```

Write results to `outputs/sae/sweep_results_010626.md`.

---

## Key Implementation Notes

- **BatchTopK pool size:** `top_k = k × 1568` passed to `BatchTopKSAE`. This is per-clip, not per-batch. Each clip's 1568 tokens are encoded through BatchTopK independently — the budget flows within a single clip, not across the 64-clip training batch.
- **Training batch size stays at 64 clips** — this is a memory/compute decision independent of the BatchTopK pool size.
- **Dead feature definition:** a feature that never activates on the held-out validation clips at end of epoch. Use the same definition across all jobs for comparability.
- **`sae.train()` / `sae.eval()`** must be called explicitly around the validation loop — do not rely on `torch.no_grad()` alone. `DictionaryLayer.get_dictionary()` in eval mode requires `_fused_dictionary` to be set, which is triggered by calling `.eval()`.
- **WandB:** log dead feature count, L0, R², MSE per epoch for all jobs. Tag each run with its job label (A–H).
- **Mean subtraction artefact:** `layer7_dim_mean.pt` must be committed to the repo (it is not a model checkpoint or activation file — it is a small preprocessing artefact, shape 768, ~3KB). Add it to `outputs/sae/` and do NOT add to `.gitignore`.

---

## Decision Criteria After Sweep

To be evaluated by Tom after results are in. Do not make architecture decisions — just record results clearly.

- If Job H (standard TopK) substantially fewer dead features than all BatchTopK jobs → BatchTopK/positional scaffold interaction confirmed as root cause
- If Job A (mean subtraction only) already resolves dead features → mean subtraction is sufficient, no other changes needed
- If Jobs B/C/D show clear α threshold → adopt that α for full training
- If Job E (zero-mask) outperforms Job A with same α → full mask is better than top-50% mask

---

## Out of Scope for This Work Package

- Per-position mean subtraction (1568×768 matrix) — on hold pending sweep results
- Changes to expansion factor (8× stays)
- Changes to training epochs (sweep is 1 epoch only)
- DFA pipeline (Stage 3) — not started yet
- Any changes to Stage 1 perturbation pipeline
