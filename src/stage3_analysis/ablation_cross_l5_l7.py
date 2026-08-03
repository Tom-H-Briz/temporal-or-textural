"""
Cross-layer ablation — L5 and L7 SAEs spliced simultaneously, ablating the
combined 11-feature set (L5's all7 + L7's all4) in one forward pass.

DFAEngine only supports splicing one layer at a time (one _hook_handle, one
SAE) — used by ~19 other callers for that exact single-layer behavior, so
this is a standalone script rather than an extension to it, registering two
independent forward hooks directly (same splice math as DFAEngine._splice_hook,
generalized to two layers).

Two conditions only, R condition only (per Tom, 01/08/26 — cleaner signal,
no shuffle confound):
  baseline — both SAEs spliced, nothing ablated (reconstruction-only; isolates
             SAE lossy-ness from the ablation effect itself)
  ablated  — both SAEs spliced, L5's 7 zeroed at layer 5, L7's 4 zeroed at
             layer 7, in the same forward pass

Clip source: SL manifest (same as dfa_mass_delta_vm.py). R-correctness is
recomputed fresh under the dual-spliced baseline, not reused from either
single-layer roster — splicing two SAEs at once can shift borderline clips.

Outputs: outputs/analysis/scaffold_ablation/ablation_cross_l5_l7_{dataset}.parquet

Usage:
    uv run python src/stage3_analysis/ablation_cross_l5_l7.py
    uv run python src/stage3_analysis/ablation_cross_l5_l7.py --dataset kinetics400
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import CHECKPOINT_REGISTRY, FRAME_SAMPLERS, MODEL_REGISTRY, resolve_sae_checkpoint
from sae import BatchTopKSAE
from stage3_analysis.dfa_engine import _preprocess_clip
from stage3_analysis.dfa_mass_delta_vm import build_sl_label_map, load_clips

L5_INDICES = [358, 449, 917, 2093, 3516, 3938, 5004]
L7_INDICES = [3347, 5165, 6021, 6032]

CFG = {
    "model_flag":      "videomae",
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "manifest_path":   str(ROOT / "outputs/Laura_SL/manifest_SL_subset.json"),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "k400_manifest_path": str(ROOT / "outputs/Laura_SL/k400_manifest_SL_subset.json"),
    "k400_sl_csv_path": str(ROOT / "outputs/Laura_SL/k400_sl_class_mapping.csv"),
    "out_dir":         ROOT / "outputs/analysis/scaffold_ablation",
}


def make_splice_hook(sae, dim_mean, cls_offset, state):
    """Mirrors DFAEngine._splice_hook's encode-ablate-decode math (forward-only,
    no backward needed for pure ablation). `state` is a mutable
    {"ablate_indices": [...]} dict toggled per forward call by the caller."""
    def hook(module, input, output):
        hidden  = output[0] if isinstance(output, tuple) else output
        cls     = hidden[:, :cls_offset]
        patches = hidden[:, cls_offset:]
        B, T, D = patches.shape
        tokens  = (patches.reshape(B * T, D) - dim_mean).float()
        with torch.no_grad():
            _, z = sae.encode(tokens)
            if state["ablate_indices"]:
                z = z.clone()
                z[:, state["ablate_indices"]] = 0.0
            recon = sae.decode(z)
        recon = (recon + dim_mean).to(hidden.dtype).reshape(B, T, D)
        out = torch.cat([cls, recon], dim=1) if cls_offset else recon
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook


def load_model_and_splices(cfg: dict, dataset_name: str):
    """Loads VideoMAE once, both SAEs (L5, L7), registers both hooks —
    DFAEngine.__enter__'s per-SAE loading (checkpoint, dim_mean, running-
    threshold warmup, freeze) repeated for two layers instead of one."""
    model_cfg  = MODEL_REGISTRY[cfg["model_flag"]]
    device     = cfg["device"]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], dataset_name)]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    states, handles = {}, []
    for layer in [5, 7]:
        resolved = resolve_sae_checkpoint(cfg["model_flag"], layer, dataset_name=dataset_name, sae_k=64, job_label="7ep")
        dim_mean = torch.load(resolved["dim_mean_path"], weights_only=True).to(device)
        ckpt = torch.load(resolved["sae_path"], weights_only=True, map_location=device)
        state_dict = ckpt["sae_state_dict"] if "sae_state_dict" in ckpt else ckpt
        nb_concepts = state_dict["dictionary._weights"].shape[0]
        sae_k = ckpt.get("sae_k") or resolved["sae_k"]
        top_k = sae_k * model_cfg["num_patch_tokens"]

        sae = BatchTopKSAE(input_shape=model_cfg["hidden_dim"], nb_concepts=nb_concepts,
                           top_k=top_k, device=device)
        sae.load_state_dict(state_dict)
        sae.train()
        dummy = torch.zeros(model_cfg["num_patch_tokens"], model_cfg["hidden_dim"], device=device)
        with torch.no_grad():
            sae.encode((dummy - dim_mean).float())
        for p in sae.parameters():
            p.requires_grad_(False)
        sae.eval()

        state = {"ablate_indices": []}
        hook  = make_splice_hook(sae, dim_mean, model_cfg["cls_offset"], state)
        handle = model_cfg["layer_getter"](model, layer).register_forward_hook(hook)
        states[layer] = state
        handles.append(handle)

    return model, processor, model_cfg["num_frames"], states, handles


def run_forward(model, pixel_values: torch.Tensor, correct_class_idx: int) -> tuple[float, int, bool]:
    with torch.no_grad():
        output = model(pixel_values=pixel_values)
    logits = output.logits.squeeze(0)
    predicted = int(logits.argmax().item())
    return float(logits[correct_class_idx].item()), predicted, predicted == correct_class_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ssv2", "kinetics400"], default="ssv2")
    args = parser.parse_args()

    frame_sampler = FRAME_SAMPLERS[args.dataset]
    sl_map = build_sl_label_map(CFG, args.dataset)
    clips  = load_clips(CFG, args.dataset)
    model, processor, num_frames, states, handles = load_model_and_splices(CFG, args.dataset)

    rows = []
    try:
        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            pixel_values = _preprocess_clip(clip_path, num_frames, processor, CFG["device"],
                                            frame_sampler=frame_sampler)
            states[5]["ablate_indices"] = []
            states[7]["ablate_indices"] = []
            base_logit, base_pred, base_correct = run_forward(model, pixel_values, class_id)
            if not base_correct:
                continue
            states[5]["ablate_indices"] = L5_INDICES
            states[7]["ablate_indices"] = L7_INDICES
            abl_logit, abl_pred, abl_correct = run_forward(model, pixel_values, class_id)
            rows.append({
                "clip_id": clip_id, "class_id": class_id,
                "sl_label": sl_map.get(class_id, "unlabelled"),
                "baseline_logit": base_logit, "ablated_logit": abl_logit,
                "delta": base_logit - abl_logit,
                "predicted_class_ablated": abl_pred, "correct_ablated": abl_correct,
            })
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(clips)}] R-correct so far: {len(rows)}")
    finally:
        for h in handles:
            h.remove()

    df = pd.DataFrame(rows)
    out_dir = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ablation_cross_l5_l7_{args.dataset}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  {len(df):,} rows → {out_path}")


if __name__ == "__main__":
    main()
