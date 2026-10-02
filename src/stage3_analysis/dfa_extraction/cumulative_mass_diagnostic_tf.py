"""
Cumulative mass diagnostic for TimeSformer — per-layer, SLURM array driven.

SSv2 (default) and K400 (DATASET_NAME=kinetics400): K400 runs over the SL
manifest with the model-aware frame sampler; output suffix is dataset-tokened.

Outputs (per layer):
  outputs/analysis/cumulative_mass_diagnostic_tf_l{N}_k64_x8.parquet            (ssv2)
  outputs/analysis/cumulative_mass_diagnostic_tf_kinetics400_l{N}.parquet       (k400)
"""

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import MODEL_REGISTRY, _strip_brackets, get_frame_sampler, load_metadata
from ToT_utils import load_clips_kinetics as tot_load_clips_kinetics
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_flag":              "timesformer",
    "dataset_name":            os.environ.get("DATASET_NAME", "ssv2"),
    "layer":                   int(os.environ.get("SAE_LAYER", 7)),
    "sae_k":                   64,
    "sae_expansion":           8,
    "device":                  "cuda" if torch.cuda.is_available() else "cpu",
    "activity_threshold_frac": 0.01,
    "mass_thresholds":         [0.50, 0.80, 0.90, 0.95],
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "k400_manifest_path": str(ROOT / "outputs/Laura_SL/k400_manifest_SL_subset.json"),
    "output_dir":      str(ROOT / "outputs/analysis"),
    "max_clips":       None,
}

N_TOKENS = MODEL_REGISTRY["timesformer"]["num_patch_tokens"]


def _resolve_cfg(layer: int, k: int, expansion: int, dataset_name: str) -> dict:
    sae_dir = ROOT / "outputs" / "sae"
    if dataset_name == "ssv2":
        # Legacy dataset-less TF naming (pre dataset-token scheme).
        matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly 1 TF best checkpoint for layer {layer}, found {len(matches)}: {matches}"
            )
        dim_mean = sae_dir / f"tf_layer{layer}_dim_mean.pt"
        suffix   = f"_tf_l{layer}_k{k}_x{expansion}"
    else:
        # K400: current dataset-tokened naming (train_sae.py job7ep convention).
        sae_path = sae_dir / f"sae_tf_{dataset_name}_k{k}_x{expansion}_l{layer}_job7ep_best.pt"
        if not sae_path.exists():
            raise FileNotFoundError(f"TF-K400 SAE not found: {sae_path}")
        matches  = [sae_path]
        dim_mean = sae_dir / f"tf_{dataset_name}_layer{layer}_dim_mean.pt"
        suffix   = f"_tf_{dataset_name}_l{layer}"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {
        "sae_path":      str(matches[0]),
        "dim_mean_path": str(dim_mean),
        "output_suffix": suffix,
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
    resolved = _resolve_cfg(CFG["layer"], CFG["sae_k"], CFG["sae_expansion"], CFG["dataset_name"])
    cfg      = {**CFG, **resolved}
    log.info(f"Layer {cfg['layer']}  dataset={cfg['dataset_name']}  checkpoint={Path(cfg['sae_path']).name}")
    log.info(f"Device: {cfg['device']}")

    frame_sampler = get_frame_sampler(cfg["dataset_name"], MODEL_REGISTRY["timesformer"])
    if cfg["dataset_name"] == "kinetics400":
        # SL-manifest clips, label2id resolved for the TF checkpoint — same population
        # the DFA mass-delta run covers (ToT_utils.load_clips_kinetics).
        clips = [(cid, cls, p, str(cls)) for cid, cls, p in
                 tot_load_clips_kinetics(cfg["k400_manifest_path"], cfg["video_dir"], "timesformer")]
    else:
        label_map, clips_meta, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
        video_dir = Path(cfg["video_dir"])
        clips = []
        for c in clips_meta:
            template = _strip_brackets(c["template"])
            if template not in label_map:
                continue
            clips.append((str(c["id"]), label_map[template], video_dir / f"{c['id']}.webm", template))

    records, n_processed, n_skipped = [], 0, 0
    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   dataset_name=cfg["dataset_name"]) as engine:
        for clip_id, class_id, clip_path, class_name in (clips[:cfg["max_clips"]] if cfg["max_clips"] else clips):
            result = engine.run(clip_path, class_id, frame_sampler=frame_sampler)
            if not result.correct:
                n_skipped += 1
                continue
            row = process_clip(result, clip_id, class_id, class_name, cfg)
            if row is not None:
                records.append(row)
            n_processed += 1
            if n_processed % 100 == 0:
                log.info(f"[{n_processed}] {class_name!r:45s} n90={records[-1].get('n90','?')}")

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
