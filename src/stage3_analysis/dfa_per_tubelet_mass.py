"""
Per-tubelet DFA mass extraction — VM, R / C1 / A, 32 SL classes.

Token ordering: temporal-major (token 0–195 = tubelet 0, 196–391 = tubelet 1, …).
Confirmed in feature_vis_group_vm.py:176 — reshape(num_tubelets, 14, 14).

Aggregates online per class — no per-clip tensors stored.

Outputs (outputs/analysis/dfa_per_tubelet_mass/):
  tubelet_position_lock.parquet     — long format: class×feature×tubelet×condition
  position_lock_scores.csv          — per class×feature×condition: top tubelet share

Usage:
    uv run python src/stage3_analysis/dfa_per_tubelet_mass.py
"""

import av
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(ROOT / "notebooks"))

from perturbationA import apply_midpoint_frame
from ToT_utils import MODEL_REGISTRY, _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

NUM_TUBELETS = 8

CFG = {
    "model_flag":      "videomae",
    "layer":           7,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/dfa_per_tubelet_mass"),
}


def _resolve_cfg(cfg: dict) -> dict:
    sae_path = ROOT / "outputs" / "sae" / "sae_layer7_job64.pt"
    dim_mean = ROOT / "outputs" / "sae" / "layer7_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"VM SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": 64}


def load_clips(cfg: dict) -> list[tuple[str, int, Path]]:
    label_map, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    result = []
    for c in clips:
        cid = label_map.get(_strip_brackets(c["template"]))
        if cid not in DFA_CLASSES:
            continue
        path = video_dir / f"{c['id']}.webm"
        if path.exists():
            result.append((str(c["id"]), cid, path))
    log.info(f"  {len(result):,} clips across {len(DFA_CLASSES)} DFA classes")
    return result


def preprocess_c1(clip_path: Path, clip_id: str, num_frames: int,
                  processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n       = len(frames)
    idx     = torch.linspace(0, n - 1, num_frames).long().tolist()
    sampled = [frames[i] for i in idx]
    pairs   = [(sampled[i], sampled[i + 1]) for i in range(0, num_frames, 2)]
    order   = np.random.default_rng(int(clip_id) % 2**32).permutation(len(pairs)).tolist()
    result  = [f for i in order for f in pairs[i]]
    return processor(result, return_tensors="pt")["pixel_values"].to(device)


def preprocess_a(clip_path: Path, num_frames: int,
                 processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_midpoint_frame(frames)
    n      = len(frames)
    idx    = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def save_outputs(
    running_abs:    dict,
    running_signed: dict,
    running_count:  dict,
    sl_map:         dict,
    out_dir:        Path,
) -> None:
    conditions = ["R", "C1", "A"]
    rows_lock, rows_score = [], []

    for class_id in sorted(running_abs):
        n     = running_count[class_id]
        label = sl_map.get(class_id, "unlabelled")
        for cond in conditions:
            mean_abs    = (running_abs[class_id][cond]    / n).numpy()  # (8, dict_size)
            mean_signed = (running_signed[class_id][cond] / n).numpy()

            for feat in range(mean_abs.shape[1]):
                feat_abs = mean_abs[:, feat]        # (8,)
                feat_sgn = mean_signed[:, feat]

                # long format rows
                for t in range(NUM_TUBELETS):
                    rows_lock.append({
                        "class_id":    class_id,
                        "sl_label":    label,
                        "feature_idx": feat,
                        "tubelet_idx": t,
                        "condition":   cond,
                        "mean_abs":    float(feat_abs[t]),
                        "mean_signed": float(feat_sgn[t]),
                        "n_clips":     n,
                    })

                # position-lock score
                total = feat_abs.sum()
                if total > 0:
                    top_t     = int(feat_abs.argmax())
                    top_share = float(feat_abs[top_t] / total)
                else:
                    top_t, top_share = 0, 0.0
                rows_score.append({
                    "class_id":         class_id,
                    "feature_idx":      feat,
                    "condition":        cond,
                    "top_tubelet_idx":  top_t,
                    "top_tubelet_share": top_share,
                    "n_clips":          n,
                })

    log.info(f"  Writing parquet ({len(rows_lock):,} rows)…")
    df_lock = pd.DataFrame(rows_lock)
    table   = pa.Table.from_pandas(df_lock, preserve_index=False)
    pq.write_table(table, str(out_dir / "tubelet_position_lock.parquet"))
    log.info(f"  Parquet → {out_dir / 'tubelet_position_lock.parquet'}")

    df_score = pd.DataFrame(rows_score)
    df_score.to_csv(out_dir / "position_lock_scores.csv", index=False)
    log.info(f"  CSV → {out_dir / 'position_lock_scores.csv'}")


def main() -> None:
    resolved = _resolve_cfg(CFG)
    cfg      = {**CFG, **resolved}

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = {int(r["class_id"]): r["category"]
              for _, r in pd.read_csv(cfg["sl_csv_path"]).iterrows()}

    clips      = load_clips(cfg)
    dict_size  = 6144

    running_abs    = defaultdict(lambda: {c: torch.zeros(NUM_TUBELETS, dict_size) for c in ["R","C1","A"]})
    running_signed = defaultdict(lambda: {c: torch.zeros(NUM_TUBELETS, dict_size) for c in ["R","C1","A"]})
    running_count  = defaultdict(int)

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=cfg["sae_k"]) as engine:

        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            try:
                r_result = engine.run(clip_path, class_id, return_per_tubelet=True)
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}"); continue

            if not r_result.correct:
                continue

            try:
                pv_c1    = preprocess_c1(clip_path, clip_id, engine._num_frames,
                                         engine._processor, cfg["device"])
                pv_a     = preprocess_a(clip_path, engine._num_frames,
                                        engine._processor, cfg["device"])
                c1_result = engine.run_pixels(pv_c1, class_id, return_per_tubelet=True)
                a_result  = engine.run_pixels(pv_a,  class_id, return_per_tubelet=True)
            except Exception as exc:
                log.warning(f"SKIP_PERT {clip_id}: {exc}"); continue

            running_abs[class_id]["R"]  += r_result.per_tubelet_abs
            running_abs[class_id]["C1"] += c1_result.per_tubelet_abs
            running_abs[class_id]["A"]  += a_result.per_tubelet_abs
            running_signed[class_id]["R"]  += r_result.per_tubelet_signed
            running_signed[class_id]["C1"] += c1_result.per_tubelet_signed
            running_signed[class_id]["A"]  += a_result.per_tubelet_signed
            running_count[class_id] += 1

            if (i + 1) % 100 == 0:
                log.info(f"[{i+1}/{len(clips)}] R-correct so far: {sum(running_count.values())}")

    log.info(f"Done — {sum(running_count.values())} R-correct clips across {len(running_count)} classes")
    save_outputs(running_abs, running_signed, running_count, sl_map, out_dir)


if __name__ == "__main__":
    main()
