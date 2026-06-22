"""
Comprehensive TF DFA attribution pass — 32 SL classes, layers [3,5,7,9].

One parquet per layer: dfa_tf_k64_l{LAYER}_sl32_{date}.parquet
Per-clip rows: metadata + scalars + sparse top-200 DFA features.
Full 6144-dim vectors are NOT persisted.

Usage:
    SAE_LAYER=7 uv run python src/stage3_analysis/dfa_extraction_tf.py
"""

import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

_LAYER = int(os.environ.get("SAE_LAYER", 7))

CFG = {
    "model_flag":      "timesformer",
    "layer":           _LAYER,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "output_dir":      str(ROOT / "outputs/dfa"),
}


def _resolve_cfg(cfg: dict) -> dict:
    sae_dir = ROOT / "outputs" / "sae"
    layer   = cfg["layer"]
    matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 TF checkpoint for layer {layer}, found: {matches}")
    ckpt_path = matches[0]
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae_k     = ckpt.get("sae_k")
    if sae_k is None:
        raise ValueError(f"sae_k not in checkpoint {ckpt_path} — run patch_checkpoints_add_sae_k.py first")
    dim_mean  = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    date = datetime.date.today().strftime("%d%m%y")
    return {
        "sae_path":      str(ckpt_path),
        "dim_mean_path": str(dim_mean),
        "sae_k":         sae_k,
        "out_name":      f"dfa_tf_k{sae_k}_l{layer}_sl32_{date}",
    }


def _compute_nx(cumsum: np.ndarray, total: float, thresh: float) -> int:
    return int(np.searchsorted(cumsum, thresh * total, side="left")) + 1


def _compute_gini(values: np.ndarray) -> float:
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    x     = np.sort(values)
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * x).sum()) / (n * x.sum()) - (n + 1) / n)


def compute_scalars(result, layer: int) -> dict:
    pfs = result.per_feature_summary.numpy()    # (6144,) abs DFA
    sfs = result.signed_feature_summary.numpy() # (6144,) signed DFA
    tfc = result.token_fire_counts.numpy()      # (6144,) per-feature token fire count

    n_active = int((tfc > 0).sum())
    sorted_abs = np.sort(pfs)[::-1]
    cumsum     = np.cumsum(sorted_abs)
    total      = float(pfs.sum())

    pos_mass    = np.clip(sfs, 0, None)
    pos_sorted  = np.sort(pos_mass)[::-1]
    pos_cumsum  = np.cumsum(pos_sorted)
    pos_total   = float(pos_mass.sum())
    signed_n90  = _compute_nx(pos_cumsum, pos_total, 0.9) if pos_total > 0 else 0
    sup_frac    = float(np.abs(sfs[sfs < 0]).sum()) / total if total > 0 else 0.0

    return {
        "layer":            layer,
        "predicted_class":  result.predicted_class,
        "correct":          result.correct,
        "logit_margin":     float(result.logit_margin),
        "softmax_entropy":  float(result.entropy_normalised),
        "n_active":         n_active,
        "n50":  _compute_nx(cumsum, total, 0.50) if total > 0 else 0,
        "n80":  _compute_nx(cumsum, total, 0.80) if total > 0 else 0,
        "n90":  _compute_nx(cumsum, total, 0.90) if total > 0 else 0,
        "n95":  _compute_nx(cumsum, total, 0.95) if total > 0 else 0,
        "gini":             _compute_gini(pfs),
        "suppressor_frac":  sup_frac,
        "signed_n90":       signed_n90,
    }


def compute_sparse(result) -> dict:
    pfs = result.per_feature_summary.numpy()
    sfs = result.signed_feature_summary.numpy()
    tfc = result.token_fire_counts.numpy()

    n_active   = int((tfc > 0).sum())
    k          = min(n_active, 200)
    sorted_idx = np.argsort(pfs)[::-1]
    real_idx   = sorted_idx[:k].astype(np.int32)

    idx    = np.full(200, -1,  dtype=np.int32)
    abs_v  = np.zeros(200,     dtype=np.float32)
    sign_v = np.zeros(200,     dtype=np.float32)
    idx[:k]    = real_idx
    abs_v[:k]  = pfs[real_idx]
    sign_v[:k] = sfs[real_idx]

    return {"top200_indices": idx, "top200_dfa_abs": abs_v, "top200_dfa_signed": sign_v}


def load_clips(cfg: dict) -> list[tuple[str, int]]:
    label_map, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    result = []
    for c in clips:
        cid = label_map.get(_strip_brackets(c["template"]))
        if cid not in DFA_CLASSES:
            continue
        path = video_dir / f"{c['id']}.webm"
        if path.exists():
            result.append((str(c["id"]), cid))
    log.info(f"  {len(result):,} clips across {len(DFA_CLASSES)} DFA classes")
    return result


def save_parquet(records: list[dict], out_path: Path) -> None:
    def arr(key, pa_type):
        return pa.array([r[key] for r in records], type=pa_type)
    def list_col(key, pa_type):
        return pa.array([r[key].tolist() for r in records], type=pa.list_(pa_type))

    table = pa.table({
        "clip_id":          arr("clip_id",         pa.string()),
        "class_id":         arr("class_id",        pa.int32()),
        "predicted_class":  arr("predicted_class", pa.int32()),
        "correct":          arr("correct",         pa.bool_()),
        "layer":            arr("layer",           pa.int32()),
        "logit_margin":     arr("logit_margin",    pa.float32()),
        "softmax_entropy":  arr("softmax_entropy", pa.float32()),
        "n_active":         arr("n_active",        pa.int32()),
        "n50":              arr("n50",             pa.int32()),
        "n80":              arr("n80",             pa.int32()),
        "n90":              arr("n90",             pa.int32()),
        "n95":              arr("n95",             pa.int32()),
        "gini":             arr("gini",            pa.float32()),
        "suppressor_frac":  arr("suppressor_frac", pa.float32()),
        "signed_n90":       arr("signed_n90",      pa.int32()),
        "top200_indices":   list_col("top200_indices",   pa.int32()),
        "top200_dfa_abs":   list_col("top200_dfa_abs",   pa.float32()),
        "top200_dfa_signed":list_col("top200_dfa_signed",pa.float32()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    log.info(f"Saved → {out_path}  ({len(records)} rows)")


def main() -> None:
    resolved = _resolve_cfg(CFG)
    cfg      = {**CFG, **resolved}
    log.info(f"Layer {cfg['layer']}  checkpoint={Path(cfg['sae_path']).name}")

    out_dir   = Path(cfg["output_dir"])
    out_path  = out_dir / f"{cfg['out_name']}.parquet"
    log_path  = out_dir / f"{cfg['out_name']}.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    clips   = load_clips(cfg)
    records = []

    with open(log_path, "w") as skip_log, \
         DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"]) as engine:
        for i, (clip_id, class_id) in enumerate(clips):
            clip_path = Path(cfg["video_dir"]) / f"{clip_id}.webm"
            try:
                result = engine.run(clip_path, class_id)
            except Exception as exc:
                msg = f"SKIP {clip_id} class={class_id}: {exc}"
                log.warning(msg)
                skip_log.write(msg + "\n")
                continue

            row = {"clip_id": clip_id, "class_id": class_id,
                   **compute_scalars(result, cfg["layer"]),
                   **compute_sparse(result)}
            records.append(row)
            if (i + 1) % 50 == 0:
                log.info(f"[{i+1}/{len(clips)}] clip={clip_id} class={class_id} "
                         f"correct={result.correct} n90={row['n90']}")

    log.info(f"Done: {len(records)} rows written, {len(clips)-len(records)} skipped")
    save_parquet(records, out_path)


if __name__ == "__main__":
    main()
