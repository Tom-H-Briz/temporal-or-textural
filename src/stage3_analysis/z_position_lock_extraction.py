"""
Activation-based position lock extraction — z from get_z(), no backward pass.

For each clip, computes per-tubelet SAE activation (z, not DFA) for R, C1, A.
Accumulates per-clip share statistics per class × condition — same corrected
metric as fixed dfa_per_tubelet_mass.py (share per clip then averaged).

No R-correct gate — answers "where does this feature fire", not causal contribution.

Token ordering: temporal-major — confirmed at feature_vis_group_vm.py:176.
tubelet_idx = token_idx // 196

Output: outputs/analysis/z_position_lock/z_position_lock_scores.csv

Usage:
    uv run python src/stage3_analysis/z_position_lock_extraction.py
"""

import av
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(ROOT / "notebooks"))

from perturbationA import apply_midpoint_frame
from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

NUM_TUBELETS = 8
DICT_SIZE    = 6144
CONDITIONS   = ["R", "C1", "A"]

CFG = {
    "model_flag":      "videomae",
    "layer":           7,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/z_position_lock"),
}


def _resolve_cfg() -> dict:
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
    log.info(f"  {len(result):,} clips across {len(DFA_CLASSES)} classes")
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


def accumulate(tubelet_z: torch.Tensor, class_id: int, cond: str,
               running_share_sum: dict, tubelet_occurrence: dict, running_abs_sum: dict) -> None:
    """Per-clip accumulation with active-feature mask."""
    col_sum            = tubelet_z.sum(dim=0)
    active             = col_sum > 1e-8
    col_max, col_argmax = tubelet_z.max(dim=0)
    running_share_sum[class_id][cond]  += col_max / col_sum.clamp(min=1e-10)
    running_abs_sum[class_id][cond]    += tubelet_z
    tubelet_occurrence[class_id][cond].scatter_add_(
        0, (col_argmax * active.long()).unsqueeze(0), active.float().unsqueeze(0)
    )


def main() -> None:
    resolved = _resolve_cfg()
    cfg      = CFG
    out_dir  = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = {int(r["class_id"]): r["category"]
              for _, r in pd.read_csv(cfg["sl_csv_path"]).iterrows()}
    clips  = load_clips(cfg)

    running_share_sum  = defaultdict(lambda: {c: torch.zeros(DICT_SIZE)              for c in CONDITIONS})
    tubelet_occurrence = defaultdict(lambda: {c: torch.zeros(NUM_TUBELETS, DICT_SIZE) for c in CONDITIONS})
    running_abs_sum    = defaultdict(lambda: {c: torch.zeros(NUM_TUBELETS, DICT_SIZE) for c in CONDITIONS})
    running_count      = defaultdict(int)

    log.info(f"Scanning {len(clips):,} clips for R/C1/A (get_z, forward pass only)…")
    with DFAEngine(cfg["model_flag"], resolved["sae_path"], resolved["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=resolved["sae_k"]) as engine:

        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            try:
                z_r = engine.get_z(clip_path)
                pv_c1 = preprocess_c1(clip_path, clip_id, engine._num_frames,
                                      engine._processor, cfg["device"])
                pv_a  = preprocess_a(clip_path, engine._num_frames,
                                     engine._processor, cfg["device"])
                z_c1 = engine.get_z_pixels(pv_c1)
                z_a  = engine.get_z_pixels(pv_a)
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}"); continue

            for z, cond in [(z_r, "R"), (z_c1, "C1"), (z_a, "A")]:
                tubelet_z = z.reshape(NUM_TUBELETS, 196, DICT_SIZE).sum(dim=1).cpu()
                accumulate(tubelet_z, class_id, cond,
                           running_share_sum, tubelet_occurrence, running_abs_sum)
            running_count[class_id] += 1

            if (i + 1) % 500 == 0:
                log.info(f"  [{i+1}/{len(clips)}] scanned")

    log.info("Building output CSV…")
    rows = []
    for class_id in sorted(running_count):
        n     = running_count[class_id]
        label = sl_map.get(class_id, "unlabelled")

        stats = {}
        for cond in CONDITIONS:
            mean_share = (running_share_sum[class_id][cond] / n).numpy()
            mean_abs   = (running_abs_sum[class_id][cond]   / n).numpy()
            occ        = tubelet_occurrence[class_id][cond].numpy()
            stats[cond] = {
                "mean_share": mean_share,
                "mode_t":     occ.argmax(axis=0),
                "frac_mode":  occ.max(axis=0) / n,
                "total_abs":  mean_abs.sum(axis=0),
                "top_abs":    mean_abs.max(axis=0),
            }

        pos_consistent = (
            (stats["R"]["mode_t"] == stats["C1"]["mode_t"]) &
            (stats["R"]["mode_t"] == stats["A"]["mode_t"])
        )

        for feat in range(DICT_SIZE):
            row = {
                "class_id":      class_id,
                "sl_label":      label,
                "feature_idx":   feat,
                "n_clips":       n,
                "total_abs_R":   float(stats["R"]["total_abs"][feat]),
                "top_abs_R":     float(stats["R"]["top_abs"][feat]),
                "pos_consistent": bool(pos_consistent[feat]),
            }
            for cond in CONDITIONS:
                row[f"mean_per_clip_share_{cond}"]        = float(stats[cond]["mean_share"][feat])
                row[f"mode_tubelet_{cond}"]               = int(stats[cond]["mode_t"][feat])
                row[f"frac_clips_matching_mode_{cond}"]   = float(stats[cond]["frac_mode"][feat])
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = out_dir / "z_position_lock_scores.csv"
    df.to_csv(out_path, index=False)
    log.info(f"  → {out_path}  ({len(df):,} rows)")
    log.info("Done.")


if __name__ == "__main__":
    main()
