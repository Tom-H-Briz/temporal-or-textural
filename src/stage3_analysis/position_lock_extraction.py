"""
Combined DFA + raw-activation position-lock extraction — one per-clip pass.

Replaces dfa_per_tubelet_mass.py and z_position_lock_extraction.py: both scripts
ran a separate forward pass per clip to recompute the same per-tubelet SAE
activation that DFAEngine already holds in memory before its gradient multiply
(DFA = gradient x activation). This script captures both quantities from a
single DFAEngine.run_pixels() call per condition (R / shuffle / A).

Gating differs by quantity and is preserved from the original scripts, not
changed by the merge: raw-activation aggregation has no correctness gate (fires
on every clip, per z_position_lock_extraction.py's design); DFA aggregation
only accumulates on R-correct clips (per dfa_per_tubelet_mass.py's design).

KNOWN DELTA vs historical dfa_per_tubelet_mass.py output: the activity-gate mask
(a clip where a feature never fires must not vote for tubelet 0) previously only
existed in z_position_lock_extraction.py's accumulate(); dfa_per_tubelet_mass.py's
inline accumulation had no such mask (confirmed 31/07 by reading the pre-merge
code directly). This script applies the mask to both quantities via one shared
accumulate_position_stats(), per spec — so DFA-based mode_tubelet/
frac_clips_matching_mode will differ from the old DFA CSVs for inactive features
(measured 31/07: ~67% of feature-columns are all-zero in a given clip for the
l7/k64/job7ep checkpoint — expected for a sparse top-k SAE with dict_size=6144).
When regression-checking against the historical L5/L7/L9 membership table, check
the specific named locked features, not the full table — those are high-activity
features by construction and are the ones actually expected to be unaffected.

Usage:
    uv run python src/stage3_analysis/position_lock_extraction.py --model videomae --layer 7
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import av
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

from transformers import AutoConfig

from perturbation import apply_shuffle
from perturbationA import apply_midpoint_frame
from ToT_utils import (
    CHECKPOINT_REGISTRY, FRAME_SAMPLERS, MODEL_REGISTRY, N_SPATIAL, _deterministic_seed,
    _strip_brackets, load_metadata, resolve_sae_checkpoint,
)
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
    "kinetics_labels_csv": os.environ.get(
        "KINETICS_LABELS_CSV", str(ROOT / "data/kinetics400/annotations/val.csv")
    ),
    "sl_csv_path":      str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "k400_sl_csv_path": str(ROOT / "outputs/Laura_SL/k400_sl_class_mapping.csv"),
    "k400_manifest_path": str(ROOT / "outputs/Laura_SL/k400_manifest_SL_subset.json"),
    "output_dir":      str(ROOT / "outputs/analysis/position_lock"),
}


def load_clips_ssv2(cfg: dict) -> list[tuple[str, int, Path]]:
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
    log.info(f"  {len(result):,} clips across {len(DFA_CLASSES)} SSv2 DFA classes")
    return result


def load_clips_kinetics(cfg: dict, model_flag: str) -> list[tuple[str, int, Path]]:
    """K400 population is the static manifest (outputs/Laura_SL/k400_manifest_SL_subset.json,
    built once by notebooks/build_k400_sl_manifest.py) rather than re-derived per run —
    mirrors load_clips_ssv2's split between a fixed clip population and a runtime
    label->class_id lookup (label2id plays the same role here as SSv2's label_map)."""
    checkpoint = CHECKPOINT_REGISTRY[(model_flag, "kinetics400")]
    label2id   = AutoConfig.from_pretrained(checkpoint).label2id
    with open(cfg["k400_manifest_path"]) as f:
        manifest = json.load(f)

    video_dir = Path(cfg["video_dir"])
    result = []
    for entries in manifest.values():
        for entry in entries:
            cid  = label2id.get(entry["label"])
            path = video_dir / f"{entry['id']}.mp4"
            if cid is not None and path.exists():
                result.append((entry["id"], cid, path))
    log.info(f"  {len(result):,} clips from K400 SL manifest")
    return result


def load_clips(cfg: dict, dataset_name: str, model_flag: str) -> list[tuple[str, int, Path]]:
    if dataset_name == "ssv2":
        return load_clips_ssv2(cfg)
    return load_clips_kinetics(cfg, model_flag)


def load_sl_map(cfg: dict, dataset_name: str) -> dict[int, str]:
    if dataset_name == "ssv2":
        return {int(r["class_id"]): r["category"]
                for _, r in pd.read_csv(cfg["sl_csv_path"]).iterrows()}
    df = pd.read_csv(cfg["k400_sl_csv_path"]).dropna(subset=["matched_model_class_id"])
    return {int(r["matched_model_class_id"]): r["sl_category"] for _, r in df.iterrows()}


def preprocess_c1(clip_path: Path, clip_id: str, num_frames: int,
                  processor, device: str, frame_sampler) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    n       = len(frames)
    idx     = frame_sampler(n, num_frames)
    sampled = [frames[i] for i in idx]
    pairs   = [(sampled[i], sampled[i + 1]) for i in range(0, num_frames, 2)]
    order   = np.random.default_rng(_deterministic_seed(clip_id)).permutation(len(pairs)).tolist()
    result  = [f for i in order for f in pairs[i]]
    return processor(result, return_tensors="pt")["pixel_values"].to(device)


def preprocess_a(clip_path: Path, num_frames: int,
                 processor, device: str, frame_sampler) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_midpoint_frame(frames)
    n      = len(frames)
    idx    = frame_sampler(n, num_frames)
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def preprocess_c_tf(clip_path: Path, clip_id: str, num_frames: int,
                    processor, device: str, frame_sampler) -> torch.Tensor:
    """TF's full-frame shuffle — matches dfa_mass_delta.py's preprocess_c. TF is
    ssv2-only in this project (resolve_sae_checkpoint asserts this)."""
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, _deterministic_seed(clip_id))
    n      = len(frames)
    idx    = frame_sampler(n, num_frames)
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


SHUFFLE_PREPROCESSOR = {"videomae": preprocess_c1, "timesformer": preprocess_c_tf}
SHUFFLE_LABEL         = {"videomae": "C1",          "timesformer": "C"}


def accumulate_position_stats(values: torch.Tensor, class_id: int, cond: str,
                              running_sum: dict, running_share_sum: dict,
                              tubelet_occurrence: dict) -> None:
    """Per-clip accumulation with active-feature mask (02/07 fix) — a clip where a
    feature never fires must not vote for position 0. Shared by both the DFA-abs
    mass and the raw-activation mass; same mask, same threshold, off the same
    per-clip tensor shape (num_positions, dict_size)."""
    running_sum[class_id][cond] += values
    col_sum              = values.sum(dim=0)
    active               = col_sum > 1e-8
    col_max, col_argmax  = values.max(dim=0)
    running_share_sum[class_id][cond] += col_max / col_sum.clamp(min=1e-10)
    tubelet_occurrence[class_id][cond].scatter_add_(
        0, (col_argmax * active.long()).unsqueeze(0), active.float().unsqueeze(0)
    )


def assert_dfa_zero_where_inactive(per_position_raw: torch.Tensor, per_position_abs: torch.Tensor) -> None:
    """DFA = gradient * activation is mechanically zero wherever activation is zero
    (dfa_engine.py's z_detached and dfa_tensor share the same z). Assert it rather
    than assume, per the project's 'guards, not trust' convention."""
    inactive = per_position_raw == 0
    assert torch.all(per_position_abs[inactive] == 0), \
        "DFA nonzero at a position with zero raw activation — multiply invariant broken"


def _class_stats(sums: dict, share_sum: dict, occ: dict, count: dict,
                 class_id: int, conditions: list[str]) -> dict:
    """Per-class×condition share/mode/frac — same reduction dfa_per_tubelet_mass.py
    and z_position_lock_extraction.py each did separately for their one quantity."""
    n = count[class_id]
    out = {}
    for cond in conditions:
        mean_mass = (sums[class_id][cond] / n).numpy()          # (num_positions, dict_size)
        occ_np    = occ[class_id][cond].numpy()                 # (num_positions, dict_size)
        out[cond] = {
            "mean_mass":  mean_mass,
            "share":      (share_sum[class_id][cond] / n).numpy(),   # (dict_size,)
            "mode":       occ_np.argmax(axis=0),                     # (dict_size,)
            "frac_mode":  occ_np.max(axis=0) / n,                    # (dict_size,)
        }
    return out


def save_outputs(dfa: dict, z: dict, sl_map: dict, out_dir: Path, conditions: list[str],
                 num_positions: int, dict_size: int, position_label: str, out_suffix: str) -> None:
    """dfa = {abs, signed, share, occ, count}; z = {raw, share, occ, count} — see main().
    One combined parquet (per class x feature x position x condition) and one
    combined scores CSV (per class x feature), replacing the two prior pairs of
    outputs from dfa_per_tubelet_mass.py and z_position_lock_extraction.py."""
    rows_lock, rows_score = [], []
    class_ids = sorted(set(dfa["count"]) | set(z["count"]))

    for class_id in class_ids:
        label     = sl_map.get(class_id, "unlabelled")
        n_dfa     = dfa["count"].get(class_id, 0)
        n_z       = z["count"].get(class_id, 0)
        dfa_stats = _class_stats(dfa["abs"], dfa["share"], dfa["occ"], dfa["count"], class_id, conditions) if n_dfa else None
        dfa_sig   = {c: (dfa["signed"][class_id][c] / n_dfa).numpy() for c in conditions} if n_dfa else None
        z_stats   = _class_stats(z["raw"], z["share"], z["occ"], z["count"], class_id, conditions) if n_z else None

        for feat in range(dict_size):
            for t in range(num_positions):
                row = {"class_id": class_id, "sl_label": label, "feature_idx": feat,
                       f"{position_label}_idx": t, "n_clips_dfa": n_dfa, "n_clips_z": n_z}
                for cond in conditions:
                    row[f"mean_dfa_abs_{cond}"]    = float(dfa_stats[cond]["mean_mass"][t, feat]) if dfa_stats else 0.0
                    row[f"mean_dfa_signed_{cond}"] = float(dfa_sig[cond][t, feat]) if dfa_sig else 0.0
                    row[f"mean_raw_activation_{cond}"] = float(z_stats[cond]["mean_mass"][t, feat]) if z_stats else 0.0
                rows_lock.append(row)

        for feat in range(dict_size):
            row = {"class_id": class_id, "sl_label": label, "feature_idx": feat,
                   "n_clips_dfa": n_dfa, "n_clips_z": n_z}
            if dfa_stats:
                modes = [dfa_stats[c]["mode"][feat] for c in conditions]
                row["pos_consistent_dfa"] = bool(modes[0] == modes[1] == modes[2])
                row["total_abs_dfa_R"] = float(dfa_stats["R"]["mean_mass"][:, feat].sum())
                row["top_abs_dfa_R"]   = float(dfa_stats["R"]["mean_mass"][:, feat].max())
                for cond in conditions:
                    row[f"mean_per_clip_share_dfa_{cond}"]      = float(dfa_stats[cond]["share"][feat])
                    row[f"mode_{position_label}_dfa_{cond}"]    = int(dfa_stats[cond]["mode"][feat])
                    row[f"frac_clips_matching_mode_dfa_{cond}"] = float(dfa_stats[cond]["frac_mode"][feat])
            if z_stats:
                modes = [z_stats[c]["mode"][feat] for c in conditions]
                row["pos_consistent_z"] = bool(modes[0] == modes[1] == modes[2])
                row["total_abs_z_R"] = float(z_stats["R"]["mean_mass"][:, feat].sum())
                row["top_abs_z_R"]   = float(z_stats["R"]["mean_mass"][:, feat].max())
                for cond in conditions:
                    row[f"mean_per_clip_share_z_{cond}"]      = float(z_stats[cond]["share"][feat])
                    row[f"mode_{position_label}_z_{cond}"]    = int(z_stats[cond]["mode"][feat])
                    row[f"frac_clips_matching_mode_z_{cond}"] = float(z_stats[cond]["frac_mode"][feat])
            rows_score.append(row)

    lock_path  = out_dir / f"position_lock_{out_suffix}.parquet"
    score_path = out_dir / f"position_lock_scores_{out_suffix}.csv"
    log.info(f"  Writing parquet ({len(rows_lock):,} rows)…")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows_lock), preserve_index=False), str(lock_path))
    log.info(f"  Parquet → {lock_path}")
    pd.DataFrame(rows_score).to_csv(score_path, index=False)
    log.info(f"  CSV → {score_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["videomae", "timesformer"], required=True)
    parser.add_argument("--dataset", choices=["ssv2", "kinetics400"], default="ssv2")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--job-label", type=str, default="7ep", help="videomae only")
    parser.add_argument("--sae-k", type=int, default=64, help="videomae fallback if checkpoint lacks sae_k")
    args = parser.parse_args()

    resolved = resolve_sae_checkpoint(args.model, args.layer, dataset_name=args.dataset,
                                      sae_k=args.sae_k, job_label=args.job_label)
    cfg           = {**CFG, "model_flag": args.model, "layer": args.layer}
    frame_sampler = FRAME_SAMPLERS[args.dataset]
    dict_size     = resolved["nb_concepts"]   # checkpoint-derived, never hardcoded

    num_positions  = MODEL_REGISTRY[args.model]["num_patch_tokens"] // N_SPATIAL
    position_label = MODEL_REGISTRY[args.model]["position_label"]
    conditions     = ["R", SHUFFLE_LABEL[args.model], "A"]
    shuffle_fn     = SHUFFLE_PREPROCESSOR[args.model]
    out_suffix     = f"{args.model}_{args.dataset}_l{args.layer}_job{resolved['job_label']}_k{resolved['sae_k']}"

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = load_sl_map(cfg, args.dataset)
    clips  = load_clips(cfg, args.dataset, args.model)

    # DFA accumulators — gated on R-correct, matching dfa_per_tubelet_mass.py
    running_abs_dfa    = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_signed_dfa = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_share_dfa  = defaultdict(lambda: {c: torch.zeros(dict_size) for c in conditions})
    tubelet_occ_dfa    = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_count_dfa  = defaultdict(int)

    # raw-activation accumulators — gate-free, matching z_position_lock_extraction.py
    running_raw_z   = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_share_z = defaultdict(lambda: {c: torch.zeros(dict_size) for c in conditions})
    tubelet_occ_z   = defaultdict(lambda: {c: torch.zeros(num_positions, dict_size) for c in conditions})
    running_count_z = defaultdict(int)

    with DFAEngine(cfg["model_flag"], resolved["sae_path"], resolved["dim_mean_path"],
                   layer=args.layer, device=cfg["device"],
                   sae_k=resolved["sae_k"], dataset_name=args.dataset) as engine:

        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            try:
                r_result = engine.run(clip_path, class_id, return_per_position=True,
                                      frame_sampler=frame_sampler)
                pv_shuf = shuffle_fn(clip_path, clip_id, engine._num_frames,
                                     engine._processor, cfg["device"], frame_sampler)
                pv_a    = preprocess_a(clip_path, engine._num_frames,
                                       engine._processor, cfg["device"], frame_sampler)
                shuf_result = engine.run_pixels(pv_shuf, class_id, return_per_position=True)
                a_result    = engine.run_pixels(pv_a,    class_id, return_per_position=True)
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}"); continue

            results = dict(zip(conditions, [r_result, shuf_result, a_result]))

            for cond, result in results.items():
                assert_dfa_zero_where_inactive(result.per_position_raw, result.per_position_abs)
                accumulate_position_stats(result.per_position_raw, class_id, cond,
                                          running_raw_z, running_share_z, tubelet_occ_z)
            running_count_z[class_id] += 1

            if r_result.correct:
                for cond, result in results.items():
                    running_signed_dfa[class_id][cond] += result.per_position_signed
                    accumulate_position_stats(result.per_position_abs, class_id, cond,
                                              running_abs_dfa, running_share_dfa, tubelet_occ_dfa)
                running_count_dfa[class_id] += 1

            if (i + 1) % 100 == 0:
                log.info(f"[{i+1}/{len(clips)}] scanned, {sum(running_count_dfa.values())} R-correct")

    log.info(f"Done — {sum(running_count_z.values())} scanned, {sum(running_count_dfa.values())} R-correct")
    dfa = {"abs": running_abs_dfa, "signed": running_signed_dfa, "share": running_share_dfa,
           "occ": tubelet_occ_dfa, "count": running_count_dfa}
    z = {"raw": running_raw_z, "share": running_share_z, "occ": tubelet_occ_z, "count": running_count_z}
    save_outputs(dfa, z, sl_map, out_dir, conditions, num_positions, dict_size, position_label, out_suffix)


if __name__ == "__main__":
    main()
