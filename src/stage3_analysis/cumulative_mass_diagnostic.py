"""
Cumulative mass diagnostic over the full SSv2 validation set.

Two purposes:
1. DFA reliability — how concentrated is the causal signal per clip?
2. SAE expansion factor diagnostic — N50/N80/N90/N95 comparable across SAE sizes.

Outputs:
  outputs/analysis/cumulative_mass_diagnostic.parquet  (full validation set)
  outputs/analysis/cumulative_mass_diagnostic_dev5.csv (5-class dev set only)
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

from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_flag": "videomae",
    "sae_path": str(ROOT / "outputs/sae/sae_layer7_job128.pt"),
    "dim_mean_path": str(ROOT / "outputs/sae/layer7_dim_mean.pt"),
    "layer": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "activity_threshold_frac": 0.01,
    "mass_thresholds": [0.50, 0.80, 0.90, 0.95],
    "dev_class_ids": [93, 94, 97, 149, 150],
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "output_dir": str(ROOT / "outputs/analysis"),
    "max_clips": None   # set to e.g. 20 to do a quick smoke-test run
}

N_TOKENS = 1568


def _active_token_threshold(activity_threshold_frac: float) -> int:
    return int(N_TOKENS * activity_threshold_frac) + 1


def compute_gini(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    x = np.sort(values)           # ascending
    ranks = np.arange(1, n + 1)  # 1-indexed
    total = x.sum()
    if total == 0.0:
        return 0.0
    return float((2 * (ranks * x).sum()) / (n * total) - (n + 1) / n)


def compute_nx(cumsum: np.ndarray, total_mass: float, threshold: float) -> int:
    """Minimum number of features (sorted descending) to reach `threshold` fraction of total mass."""
    target = threshold * total_mass
    idx = int(np.searchsorted(cumsum, target, side="left"))
    return idx + 1


def compute_softmax_entropy_normalised(all_logits: torch.Tensor) -> float:
    """Entropy of softmax, normalised by log(num_classes) to [0, 1]."""
    probs = torch.softmax(all_logits.float(), dim=0)
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
    return entropy / math.log(probs.shape[0])


def process_clip(result, clip_id: str, class_id: int, class_name: str, cfg: dict) -> dict | None:
    threshold = _active_token_threshold(cfg["activity_threshold_frac"])
    active_mask = (result.token_fire_counts >= threshold).numpy()

    per_feat = result.per_feature_summary.numpy()
    signed_feat = result.signed_feature_summary.numpy()

    active_dfa = per_feat[active_mask]
    active_signed = signed_feat[active_mask]
    n_active = int(active_mask.sum())

    if n_active == 0:
        log.warning(f"Clip {clip_id} (class {class_id}): n_active=0 after activity filter — skipping")
        return None

    total_mass = float(active_dfa.sum())
    if total_mass == 0.0:
        log.warning(f"Clip {clip_id} (class {class_id}): total DFA mass=0 — skipping")
        return None

    sorted_dfa = np.sort(active_dfa)[::-1]
    cumsum = np.cumsum(sorted_dfa)

    mass_cols = {}
    threshold_labels = {0.50: "n50", 0.80: "n80", 0.90: "n90", 0.95: "n95"}
    for t in cfg["mass_thresholds"]:
        label = threshold_labels[t]
        mass_cols[label] = compute_nx(cumsum, total_mass, t)

    max_dfa = float(sorted_dfa[0])
    mean_dfa = float(active_dfa.mean())

    negative_mass = float(np.abs(active_signed[active_signed < 0]).sum())

    return {
        "clip_id": clip_id,
        "class_id": class_id,
        "class_name": class_name,
        "correct": True,
        "logit": result.correct_class_logit,
        "logit_margin": result.logit_margin,
        "softmax_entropy": compute_softmax_entropy_normalised(result.all_logits),
        "n_active": n_active,
        **mass_cols,
        "max_dfa": max_dfa,
        "mean_dfa": mean_dfa,
        "max_mean_ratio": max_dfa / mean_dfa if mean_dfa > 0 else float("nan"),
        "top1_frac": max_dfa / total_mass,
        "gini": compute_gini(active_dfa),
        "suppressor_frac": negative_mass / total_mass,
    }


def main() -> None:
    cfg = CFG
    log.info(f"Device: {cfg['device']}")

    active_threshold = _active_token_threshold(cfg["activity_threshold_frac"])
    log.info(
        f"Activity threshold: >= {active_threshold} tokens "
        f"({cfg['activity_threshold_frac']*100:.1f}% of {N_TOKENS})"
    )

    label_map, clips, id2template = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])

    log.info(f"Validation clips: {len(clips)}, classes: {len(label_map)}")

    records = []
    n_processed = 0
    n_skipped_wrong = 0
    seen_class_ids: set[int] = set()

    with DFAEngine(
        model_flag=cfg["model_flag"],
        sae_path=cfg["sae_path"],
        dim_mean_path=cfg["dim_mean_path"],
        layer=cfg["layer"],
        device=cfg["device"],
    ) as engine:
        for clip_info in clips[:cfg["max_clips"]] if cfg["max_clips"] else clips:
            clip_id = str(clip_info["id"])
            template_stripped = _strip_brackets(clip_info["template"])
            if template_stripped not in label_map:
                continue
            class_id = label_map[template_stripped]
            class_name = template_stripped
            clip_path = video_dir / f"{clip_id}.webm"

            result = engine.run(clip_path, class_id)

            if not result.correct:
                n_skipped_wrong += 1
                continue

            seen_class_ids.add(class_id)
            row = process_clip(result, clip_id, class_id, class_name, cfg)
            if row is not None:
                records.append(row)

            n_processed += 1
            if n_processed % 100 == 0:
                last = records[-1] if records else {}
                log.info(
                    f"[{n_processed}] class={last.get('class_name','?')!r:40s} "
                    f"logit={last.get('logit', float('nan')):6.2f}  "
                    f"n90={last.get('n90', '?')}"
                )

    all_class_ids = set(label_map.values())
    missing = all_class_ids - seen_class_ids
    if missing:
        for cid in sorted(missing):
            log.warning(f"Class {cid} ({id2template.get(cid,'?')}): zero correctly-classified clips")

    log.info(f"Done. Processed={n_processed}, skipped_wrong={n_skipped_wrong}, rows={len(records)}")

    if not records:
        log.error("No records produced — check model/SAE paths and data.")
        return

    df = pd.DataFrame(records)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "cumulative_mass_diagnostic.parquet"
    df.to_parquet(parquet_path, index=False)
    log.info(f"Saved → {parquet_path}")

    dev_df = df[df["class_id"].isin(cfg["dev_class_ids"])]
    csv_path = out_dir / "cumulative_mass_diagnostic_dev5.csv"
    dev_df.to_csv(csv_path, index=False)
    log.info(f"Saved dev5 → {csv_path}  ({len(dev_df)} rows)")


if __name__ == "__main__":
    main()
