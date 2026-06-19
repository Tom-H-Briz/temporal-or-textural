"""
Cumulative mass diagnostic for TimeSformer — per-layer, SLURM array driven.

Outputs (per layer):
  outputs/analysis/cumulative_mass_diagnostic_tf_l{N}_k64_x8.parquet
"""

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import MODEL_REGISTRY, _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_flag":              "timesformer",
    "layer":                   int(os.environ.get("SAE_LAYER", 7)),
    "device":                  "cuda" if torch.cuda.is_available() else "cpu",
    "activity_threshold_frac": 0.01,
    "mass_thresholds":         [0.50, 0.80, 0.90, 0.95],
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "output_dir":      str(ROOT / "outputs/analysis"),
    "max_clips":       None,
}

N_TOKENS = MODEL_REGISTRY["timesformer"]["num_patch_tokens"]


def _resolve_cfg(layer: int) -> dict:
    sae_dir = ROOT / "outputs" / "sae"
    matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 TF best checkpoint for layer {layer}, found {len(matches)}: {matches}"
        )
    ckpt_path = matches[0]
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state     = ckpt.get("sae_state_dict", ckpt)
    sae_k     = ckpt.get("sae_k")
    if sae_k is None:
        raise ValueError(f"sae_k not saved in checkpoint {ckpt_path} — retrain or pass manually")
    hidden_dim  = MODEL_REGISTRY["timesformer"]["hidden_dim"]
    nb_concepts = state["dictionary._weights"].shape[0]
    expansion   = nb_concepts // hidden_dim
    dim_mean    = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {
        "sae_path":      str(ckpt_path),
        "dim_mean_path": str(dim_mean),
        "output_suffix": f"_tf_l{layer}_k{sae_k}_x{expansion}",
    }


def _active_token_threshold(frac: float) -> int:
    return int(N_TOKENS * frac) + 1


def compute_gini(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    x = np.sort(values)
    ranks = np.arange(1, n + 1)
    total = x.sum()
    if total == 0.0:
        return 0.0
    return float((2 * (ranks * x).sum()) / (n * total) - (n + 1) / n)


def compute_nx(cumsum: np.ndarray, total_mass: float, threshold: float) -> int:
    return int(np.searchsorted(cumsum, threshold * total_mass, side="left")) + 1


def compute_softmax_entropy_normalised(logits: torch.Tensor) -> float:
    probs   = torch.softmax(logits.float(), dim=0)
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
    return entropy / math.log(probs.shape[0])


def process_clip(result, clip_id: str, class_id: int, class_name: str, cfg: dict) -> dict | None:
    threshold    = _active_token_threshold(cfg["activity_threshold_frac"])
    active_mask  = (result.token_fire_counts >= threshold).numpy()
    per_feat     = result.per_feature_summary.numpy()
    signed_feat  = result.signed_feature_summary.numpy()
    active_dfa   = per_feat[active_mask]
    n_active     = int(active_mask.sum())

    if n_active == 0:
        log.warning(f"Clip {clip_id}: n_active=0 — skipping")
        return None
    total_mass = float(active_dfa.sum())
    if total_mass == 0.0:
        log.warning(f"Clip {clip_id}: total DFA mass=0 — skipping")
        return None

    sorted_dfa = np.sort(active_dfa)[::-1]
    cumsum     = np.cumsum(sorted_dfa)
    labels     = {0.50: "n50", 0.80: "n80", 0.90: "n90", 0.95: "n95"}
    mass_cols  = {labels[t]: compute_nx(cumsum, total_mass, t) for t in cfg["mass_thresholds"]}
    max_dfa    = float(sorted_dfa[0])
    mean_dfa   = float(active_dfa.mean())
    neg_mass   = float(np.abs(signed_feat[active_mask][signed_feat[active_mask] < 0]).sum())

    return {
        "clip_id": clip_id, "class_id": class_id, "class_name": class_name,
        "logit": result.correct_class_logit, "logit_margin": result.logit_margin,
        "softmax_entropy": compute_softmax_entropy_normalised(result.all_logits),
        "n_active": n_active, **mass_cols,
        "max_dfa": max_dfa, "mean_dfa": mean_dfa,
        "max_mean_ratio": max_dfa / mean_dfa if mean_dfa > 0 else float("nan"),
        "top1_frac": max_dfa / total_mass, "gini": compute_gini(active_dfa),
        "suppressor_frac": neg_mass / total_mass,
    }


def main() -> None:
    resolved = _resolve_cfg(CFG["layer"])
    cfg      = {**CFG, **resolved}
    log.info(f"Layer {cfg['layer']}  checkpoint={Path(cfg['sae_path']).name}")
    log.info(f"Device: {cfg['device']}")

    label_map, clips, id2template = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])

    records, n_processed, n_skipped = [], 0, 0
    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"]) as engine:
        for clip_info in (clips[:cfg["max_clips"]] if cfg["max_clips"] else clips):
            template = _strip_brackets(clip_info["template"])
            if template not in label_map:
                continue
            clip_path = video_dir / f"{clip_info['id']}.webm"
            result    = engine.run(clip_path, label_map[template])
            if not result.correct:
                n_skipped += 1
                continue
            row = process_clip(result, str(clip_info["id"]), label_map[template], template, cfg)
            if row is not None:
                records.append(row)
            n_processed += 1
            if n_processed % 100 == 0:
                log.info(f"[{n_processed}] {template!r:45s} n90={records[-1].get('n90','?')}")

    log.info(f"Done: processed={n_processed} skipped_wrong={n_skipped} rows={len(records)}")
    if not records:
        log.error("No records — check paths.")
        return

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cumulative_mass_diagnostic{cfg['output_suffix']}.parquet"
    pd.DataFrame(records).to_parquet(out_path, index=False)
    log.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
