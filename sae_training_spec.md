# SAE Training — Specification

**Project:** Temporal or Textural? Mechanistic Interpretability of Video Transformers Using Sparse Autoencoders  
**Date:** 27 May 2026  
**Status:** Ready for implementation

---

## Overview

Train a TopK Sparse Autoencoder on the full residual stream of VideoMAE at layer 7, using real SSV2 validation clips only. The SAE learns an overcomplete dictionary of features that are more monosemantic than individual neurons, enabling downstream identification of temporal features via gradient attribution. Layer 7 is trained first to validate the full pipeline end-to-end before expanding to additional layers.

The SAE is trained on the residual stream activations of all 1568 patch tokens per clip (CLS excluded). Each token is treated as an independent training sample within a clip-level batch.

---

## Model and Hookpoint

- **Model:** `MCG-NJU/videomae-base-finetuned-ssv2` (VideoMAE ViT-Base), frozen throughout
- **Layer:** 7 (post-residual addition, full transformer block output)
- **Token sequence:** 1568 patch tokens per clip, shape `(1568, 768)` — CLS token excluded
- **Activation precision:** float16 for storage and forward pass; float32 for SAE weights

---

## Dataset Split

- **Source:** SSV2 validation set, real clips only (~24k clips)
- **Training set:** 20k clips, randomly sampled without stratification
- **Validation set:** 4k clips, randomly sampled without stratification
- **Split:** random, no class stratification required at this scale
- **Perturbed clips excluded from training** — SAE sees only unmodified video

---

## SAE Architecture

- **Type:** BatchTopK SAE (fallback to standard TopK if training is unstable)
- **Input dimension:** 768 (VideoMAE hidden dim)
- **Expansion factor:** 8×
- **Dictionary size:** 768 × 8 = 6,144 features
- **Sparsity:** k=64 active features per token (BatchTopK enforces this on average across the batch; standard TopK enforces exactly per token)
- **Auxiliary loss coefficient:** α=0.03 (dead latent recovery, per Dokme & Vishwanath 2026)

### BatchTopK vs Standard TopK

BatchTopK is preferred because the sparsity budget is allocated across a token pool rather than enforced exactly per token. This allows tokens in action-relevant regions to use more dictionary capacity where the signal is richest, rather than forcing exactly 64 features onto every static background patch.

**Pool size decision:** BatchTopK is applied at the clip level — one clip = 1568 tokens is the pool over which the global sparsity budget is enforced. This is the natural unit: the budget flows between background and action-region tokens within a single clip where spatial and temporal structure is coherent, without introducing cross-clip dependencies.

**`top_k` passed to `BatchTopKSAE`:**
```python
top_k = 64 * 1568  # = 100_352 — fixed, one clip's token budget
```

**At inference:** BatchTopKSAE switches to JumpReLU using the running threshold `θ` estimated during training (average of minimum positive activations across batches). The pool size is a training dynamic only — inference behaviour is decoupled from it. All DFA analysis and validation metrics use the running threshold.

If BatchTopK produces training instability or elevated dead feature rates, fall back to standard TopK with `k=64` per token and document the reason.

---

## Training Configuration

- **Epochs:** 5
- **Training batch size:** 64 clips — for memory management and gradient stability only. This is the number of clips loaded per gradient step on the GH200.
- **BatchTopK pool size:** 1 clip = 1568 tokens — independent of training batch size. Each clip is encoded through BatchTopK separately; the sparsity budget is allocated within a single clip's tokens, not across the 64-clip training batch.
- **Total clip passes:** 20k clips × 5 epochs = 100k (matching Dokme & Vishwanath 2026)
- **Optimiser:** Adam (standard for SAE training)
- **Precision:** float16 activations, float32 SAE weights

These are distinct concepts:
- **64 clips per gradient step** — a memory and compute decision; gradients accumulated across all 64 clips before weight update
- **1568 tokens per BatchTopK pool** — an architectural decision; each clip's tokens are encoded independently through BatchTopK before gradients are accumulated

### Streaming Pipeline

The full activation tensor is not materialised. Training uses a streaming approach:

1. Shuffle clip indices at the start of each epoch
2. Load batch of 64 clips in shuffled order
3. Forward pass through frozen VideoMAE, hook layer 7 post-residual output — shape `(64, 1568, 768)`
4. For each clip independently: pass its 1568 tokens through BatchTopK SAE encoder. The BatchTopK sparsity budget is allocated within this single clip's token pool, not across all 64 clips
5. Accumulate gradients across all 64 clips
6. Update SAE weights once per 64-clip batch
7. Repeat for all clips, then repeat for 5 epochs

---

## Validation Metrics

Computed on held-out 4k clips after each epoch and at training completion:

- **R²** — variance explained on held-out tokens. Primary reconstruction quality metric.
- **MSE** — reconstruction error on held-out tokens, complementary to R²
- **L0** — mean active features per token. Should be ~64 by construction; sanity check only
- **Dead features** — features that never activate on held-out clips. Elevated dead feature rate indicates dictionary collapse or poor initialisation
- **Feature density histogram** — distribution of activation frequency across the 6,144 features on held-out clips. Identifies ultra-dense features (polysemantic) and dead features

WandB used for experiment tracking throughout.

---

## Output

- SAE checkpoint: `outputs/sae/sae_layer_7.pt`
- Validation metrics logged to WandB per epoch
- Feature density histogram saved at training completion

---

## Temporal Feature Definition

Following Venkataramanan et al. (Chirality in Action), a feature is defined as **temporal** if its causal contribution to the classification decision is disrupted by temporal perturbation. Operationally: a feature is a temporal candidate if its DFA attribution score is significantly higher on real clips than on perturbed clips, within clips where model classification behaviour also changes. This definition is applied in Stage 3; it is recorded here as it motivates the SAE training design choices.

---

## Comparability with Dokme & Vishwanath (2026)

| Parameter | This project | Dokme & Vishwanath |
|-----------|-------------|-------------------|
| Model | VideoMAE-B | VideoMAE-B + DINOv2 |
| Dataset | SSV2 (real clips only) | SSV2 + K400 |
| Training clips | 20k | 10k |
| Epochs | 5 | 10 |
| Total clip passes | 100k | 100k |
| Expansion factor | 8× | 8× |
| k | 64 | 64 |
| α | 0.03 | 0.03 |
| Batch size (clips) | 64 | ~2.6 (compute constrained) |
| Hookpoint | Layer 7, post-residual | Layers 3, 7, 11 |
| SAE type | BatchTopK (preferred) | Standard TopK |

Batch size difference is deliberate — GH200 (96GB HBM3) allows larger batches than Dokme's A100 setup. Larger clip-level batches produce more stable gradient estimates and better dead feature recovery.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Layer | 7 only (first pass) | Dokme & Vishwanath find layer 7 most action-discriminative for VideoMAE on SSV2 (32.9% probe vs 15.9% at layer 11); validates full pipeline before expanding |
| Hookpoint | Post-residual, full patch token stream | Preserves spatiotemporal identity of each token; CLS excluded as it loses spatial and temporal traceability |
| SAE type | BatchTopK at clip level, fallback to standard TopK | BatchTopK pool = 1 clip = 1568 tokens; budget flows within clip where spatial and temporal structure is coherent. `top_k = 64 × 1568 = 100,352` passed to `BatchTopKSAE`. Inference uses running threshold (JumpReLU), decoupled from pool size. Fallback to standard TopK with k=64 if unstable |
| Training data | Real clips only | SAE learns the model's natural activation distribution; perturbed clips are inference-time analysis only |
| Dataset split | Random 20k/4k, no stratification | Scale makes stratification unnecessary; simplicity preferred |
| Total clip passes | 100k | Matches Dokme & Vishwanath for comparability |
| Batch size | 64 clips | GH200 capacity allows larger batches than prior work; more stable gradient estimates and better auxiliary loss behaviour |
| CLS exclusion | Excluded throughout | CLS-only SAE loses spatiotemporal traceability; full patch token training enables localisation of feature activations to specific tubelets and frames |
| Expansion factor | 8× (6,144 features) | Per Dokme & Vishwanath — zero dead features, strong reconstruction at this scale on VideoMAE/SSV2 |
| Contrastive objectives | Not used | Contrastive variants optimise temporal coherence across frames, not causal decision-relevance; would put a finger on the scale toward finding temporal features |

---

## Dependencies

- `overcomplete` — SAE implementation, including BatchTopK and standard TopK variants. Source: [KempnerInstitute/overcomplete](https://github.com/KempnerInstitute/overcomplete). Installed as a local dependency in `src/sae/`. All SAE architecture code sourced from this library — no custom SAE implementation.
- `torch` — VideoMAE forward pass and activation hooks
- `transformers` — VideoMAE model loading
- `wandb` — experiment tracking
- `numpy` — activation handling
- `pathlib` — path handling
-  need to consider the overcomplete libraries methods. for example "the DictionaryLayer has a quirk — get_dictionary() in eval mode requires _fused_dictionary to be set (triggered by calling .eval()). We need to be careful to call sae.train() / sae.eval() explicitly around the validation loop, not just rely on torch.no_grad()" 
---

## Explicit Non-Goals

- No contrastive or Matryoshka objectives
- No CLS token training
- No multi-layer training in this pass — layer 7 only
- No perturbed clips in training data
