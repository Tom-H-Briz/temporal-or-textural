"""
Per-position DFA mass extraction — VM + TF, R / shuffle / A, 32 SL classes.

Position = VM tubelet or TF frame (MODEL_REGISTRY[model_flag]["position_label"]).
Token-to-position gathering is model-aware — see ToT_utils.gather_by_position.
Shuffle condition is "C1" (pairwise, tubelet-aware) for VM and "C" (full shuffle,
matches dfa_mass_delta.py) for TF — not the same perturbation, see SHUFFLE_LABEL.

Aggregates online per class — no per-clip tensors stored.

Outputs (outputs/analysis/dfa_per_tubelet_mass/):
  tubelet_position_lock_{model}_l{layer}.parquet   — long format: class×feature×position×condition
  position_lock_scores_{model}_l{layer}.csv        — per class×feature×condition: top position share

Usage:
    uv run python src/stage3_analysis/dfa_per_tubelet_mass.py --model videomae --layer 7
    uv run python src/stage3_analysis/dfa_per_tubelet_mass.py --model timesformer --layer 5
"""

import argparse
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

from perturbation import apply_shuffle
from perturbationA import apply_midpoint_frame
from ToT_utils import MODEL_REGISTRY, N_SPATIAL, _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

CFG = {
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/dfa_per_tubelet_mass"),
}


def _resolve_cfg(model_flag: str, layer: int) -> dict:
    sae_dir = ROOT / "outputs" / "sae"
    if model_flag == "videomae":
        sae_path = sae_dir / f"sae_layer{layer}_job64.pt"     # job64 pinned — see brief §clarification
        dim_mean = sae_dir / f"layer{layer}_dim_mean.pt"
        sae_k    = 64
    else:  # timesformer
        matches = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected 1 TF checkpoint for layer {layer}, found: {matches}")
        sae_path = matches[0]
        sae_k    = torch.load(sae_path, map_location="cpu", weights_only=True).get("sae_k")
        dim_mean = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"{model_flag} SAE not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": sae_k}


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


def preprocess_c_tf(clip_path: Path, clip_id: str, num_frames: int,
                    processor, device: str) -> torch.Tensor:
    """TF's full-frame shuffle — matches dfa_mass_delta.py's preprocess_c. No tubelet
    pairing (TF has no tubelet structure to preserve)."""
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, int(clip_id) % 2**32)
    n      = len(frames)
    idx    = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


SHUFFLE_PREPROCESSOR = {"videomae": preprocess_c1, "timesformer": preprocess_c_tf}
SHUFFLE_LABEL         = {"videomae": "C1",          "timesformer": "C"}


def save_outputs(
    running_abs:       dict,
    running_signed:    dict,
    running_share_sum: dict,
    tubelet_occurrence: dict,
    running_count:     dict,
    sl_map:            dict,
    out_dir:           Path,
    conditions:        list[str],
    num_positions:     int,
    position_label:    str,
    out_suffix:        str,
) -> None:
    rows_lock, rows_score = [], []

    for class_id in sorted(running_abs):
        n     = running_count[class_id]
        label = sl_map.get(class_id, "unlabelled")

        # Per-condition per-clip share metrics (vectorised over all features)
        pc_share, mode_t, frac_mode = {}, {}, {}
        for cond in conditions:
            pc_share[cond] = (running_share_sum[class_id][cond] / n).numpy()  # (dict_size,)
            occ = tubelet_occurrence[class_id][cond].numpy()                   # (num_positions, dict_size)
            mode_t[cond]   = occ.argmax(axis=0)                               # (dict_size,)
            frac_mode[cond] = occ.max(axis=0) / n                             # (dict_size,)

        # Aggregate abs/signed for parquet and total_abs_R / top_abs_R
        mean_abs_R = (running_abs[class_id]["R"] / n).numpy()   # (num_positions, dict_size)
        total_abs_R = mean_abs_R.sum(axis=0)                    # (dict_size,)
        top_abs_R   = mean_abs_R.max(axis=0)                    # (dict_size,)

        for cond in conditions:
            mean_abs    = (running_abs[class_id][cond]    / n).numpy()
            mean_signed = (running_signed[class_id][cond] / n).numpy()
            for feat in range(mean_abs.shape[1]):
                for t in range(num_positions):
                    rows_lock.append({
                        "class_id":    class_id,
                        "sl_label":    label,
                        "feature_idx": feat,
                        f"{position_label}_idx": t,
                        "condition":   cond,
                        "mean_abs":    float(mean_abs[t, feat]),
                        "mean_signed": float(mean_signed[t, feat]),
                        "n_clips":     n,
                    })

        pos_consistent = (mode_t["R"] == mode_t[conditions[1]]) & (mode_t["R"] == mode_t[conditions[2]])

        for feat in range(total_abs_R.shape[0]):
            row = {
                "class_id":    class_id,
                "feature_idx": feat,
                "n_clips":     n,
                "total_abs_R": float(total_abs_R[feat]),
                "top_abs_R":   float(top_abs_R[feat]),
                "pos_consistent": bool(pos_consistent[feat]),
            }
            for cond in conditions:
                row[f"mean_per_clip_share_{cond}"]    = float(pc_share[cond][feat])
                row[f"mode_{position_label}_{cond}"]  = int(mode_t[cond][feat])
                row[f"frac_clips_matching_mode_{cond}"] = float(frac_mode[cond][feat])
            rows_score.append(row)

    lock_path  = out_dir / f"tubelet_position_lock_{out_suffix}.parquet"
    score_path = out_dir / f"position_lock_scores_{out_suffix}.csv"
    log.info(f"  Writing parquet ({len(rows_lock):,} rows)…")
    df_lock = pd.DataFrame(rows_lock)
    table   = pa.Table.from_pandas(df_lock, preserve_index=False)
    pq.write_table(table, str(lock_path))
    log.info(f"  Parquet → {lock_path}")

    df_score = pd.DataFrame(rows_score)
    df_score.to_csv(score_path, index=False)
    log.info(f"  CSV → {score_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["videomae", "timesformer"], required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    resolved = _resolve_cfg(args.model, args.layer)
    cfg      = {**CFG, "model_flag": args.model, "layer": args.layer, **resolved}

    num_positions  = MODEL_REGISTRY[args.model]["num_patch_tokens"] // N_SPATIAL
    position_label = MODEL_REGISTRY[args.model]["position_label"]
    conditions     = ["R", SHUFFLE_LABEL[args.model], "A"]
    shuffle_fn     = SHUFFLE_PREPROCESSOR[args.model]
    out_suffix     = f"{args.model}_l{args.layer}"

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = {int(r["class_id"]): r["category"]
              for _, r in pd.read_csv(cfg["sl_csv_path"]).iterrows()}

    clips      = load_clips(cfg)
    dict_size  = 6144

    running_abs    = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_signed = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    # per-clip share accumulators — fix for aggregate-mass share bias
    running_share_sum    = defaultdict(lambda: {c: torch.zeros(dict_size) for c in conditions})
    tubelet_occurrence   = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_count  = defaultdict(int)

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=cfg["sae_k"]) as engine:

        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            try:
                r_result = engine.run(clip_path, class_id, return_per_position=True)
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}"); continue

            if not r_result.correct:
                continue

            try:
                pv_shuf  = shuffle_fn(clip_path, clip_id, engine._num_frames,
                                      engine._processor, cfg["device"])
                pv_a     = preprocess_a(clip_path, engine._num_frames,
                                        engine._processor, cfg["device"])
                shuf_result = engine.run_pixels(pv_shuf, class_id, return_per_position=True)
                a_result    = engine.run_pixels(pv_a,    class_id, return_per_position=True)
            except Exception as exc:
                log.warning(f"SKIP_PERT {clip_id}: {exc}"); continue

            for cond, result in zip(conditions, [r_result, shuf_result, a_result]):
                pabs = result.per_position_abs          # (num_positions, dict_size)
                running_abs[class_id][cond]    += pabs
                running_signed[class_id][cond] += result.per_position_signed

                col_sum    = pabs.sum(dim=0).clamp(min=1e-10)  # (dict_size,)
                col_max, col_argmax = pabs.max(dim=0)           # (dict_size,) each
                running_share_sum[class_id][cond] += col_max / col_sum
                tubelet_occurrence[class_id][cond].scatter_add_(
                    0, col_argmax.unsqueeze(0), torch.ones(1, dict_size)
                )
            running_count[class_id] += 1

            if (i + 1) % 100 == 0:
                log.info(f"[{i+1}/{len(clips)}] R-correct so far: {sum(running_count.values())}")

    log.info(f"Done — {sum(running_count.values())} R-correct clips across {len(running_count)} classes")
    save_outputs(running_abs, running_signed, running_share_sum,
                 tubelet_occurrence, running_count, sl_map, out_dir,
                 conditions, num_positions, position_label, out_suffix)


if __name__ == "__main__":
    main()
