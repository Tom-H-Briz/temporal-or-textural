"""
Chiral extraction: per-clip DFA data for 10 Pushing L→R (class 93) and
10 Pushing R→L (class 94) clips, with perturbation filter conditions.

Clip selection: top 10 per class by max_mean_ratio from
cumulative_mass_diagnostic.parquet.

Conditions:
  R — real clip, full DFA (engine.run) — per_feature_summary, signed, token counts, scalars
  A — midpoint still (engine.get_z) — token_fire_counts only
  B — first/last    (engine.get_z) — token_fire_counts only

A/B fire counts are stored raw so the activity threshold can be applied
downstream when building the noise filter.

Outputs:
  outputs/analysis/chiral_extraction.parquet  (20 rows, wide format)
"""

import logging
import os
import sys
import tempfile
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

from perturbation import apply_first_last, get_clip_properties, read_frames, write_clip
from perturbationA import apply_midpoint_frame
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

CFG = {
    "model_flag": "videomae",
    "sae_path": str(ROOT / "outputs/sae/sae_layer7_job128.pt"),
    "dim_mean_path": str(ROOT / "outputs/sae/layer7_dim_mean.pt"),
    "layer": 7,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "parquet_path": str(ROOT / "outputs/analysis/cumulative_mass_diagnostic.parquet"),
    "n_clips_per_class": 10,
    "class_ids": [93, 94],
    "activity_threshold": 17,
    "output_path": str(ROOT / "outputs/analysis/chiral_extraction.parquet"),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
}


def select_clips(cfg: dict) -> list[tuple[str, int, str]]:
    diag = pd.read_parquet(cfg["parquet_path"])
    result = []
    for class_id in cfg["class_ids"]:
        subset = (
            diag[diag["class_id"] == class_id]
            .sort_values("max_mean_ratio", ascending=False)
            .head(cfg["n_clips_per_class"])
        )
        class_name = subset["class_name"].iloc[0]
        for _, row in subset.iterrows():
            result.append((str(row["clip_id"]), class_id, class_name))
        log.info(f"Class {class_id} ({class_name}): {len(subset)} clips selected")
    return result


def r_max_mean_ratio(pfs: np.ndarray, tfc: np.ndarray, threshold: int) -> float:
    active = pfs[tfc >= threshold]
    if len(active) == 0 or active.mean() == 0:
        return float("nan")
    return float(active.max() / active.mean())


def get_token_fire_counts(engine: DFAEngine, clip_path: Path) -> np.ndarray:
    z = engine.get_z(clip_path)                          # (num_tokens, dict_size)
    return (z > 0).sum(dim=0).to(torch.int32).cpu().numpy()


def run_clip(
    engine: DFAEngine,
    clip_id: str,
    class_id: int,
    class_name: str,
    video_dir: Path,
    cfg: dict,
) -> dict:
    clip_path = video_dir / f"{clip_id}.webm"

    r = engine.run(clip_path, class_id)
    r_pfs = r.per_feature_summary.numpy()
    r_sfs = r.signed_feature_summary.numpy()
    r_tfc = r.token_fire_counts.numpy()

    frames = read_frames(clip_path)
    props = get_clip_properties(clip_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        path_A = tmp / f"{clip_id}A.webm"
        write_clip(apply_midpoint_frame(frames), path_A, props)
        a_tfc = get_token_fire_counts(engine, path_A)

        path_B = tmp / f"{clip_id}B.webm"
        write_clip(apply_first_last(frames), path_B, props)
        b_tfc = get_token_fire_counts(engine, path_B)

    return {
        "clip_id": clip_id,
        "class_id": class_id,
        "class_name": class_name,
        "r_correct": r.correct,
        "r_logit": float(r.correct_class_logit),
        "r_logit_margin": float(r.logit_margin),
        "r_max_mean_ratio": r_max_mean_ratio(r_pfs, r_tfc, cfg["activity_threshold"]),
        "r_per_feature_summary": r_pfs,
        "r_signed_feature_summary": r_sfs,
        "r_token_fire_counts": r_tfc,
        "a_token_fire_counts": a_tfc,
        "b_token_fire_counts": b_tfc,
    }


def save_parquet(records: list[dict], out_path: Path) -> None:
    def list_col(key, pa_type):
        return pa.array([r[key].tolist() for r in records], type=pa.list_(pa_type))

    table = pa.table({
        "clip_id":                  pa.array([r["clip_id"] for r in records], type=pa.string()),
        "class_id":                 pa.array([r["class_id"] for r in records], type=pa.int32()),
        "class_name":               pa.array([r["class_name"] for r in records], type=pa.string()),
        "r_correct":                pa.array([r["r_correct"] for r in records], type=pa.bool_()),
        "r_logit":                  pa.array([r["r_logit"] for r in records], type=pa.float32()),
        "r_logit_margin":           pa.array([r["r_logit_margin"] for r in records], type=pa.float32()),
        "r_max_mean_ratio":         pa.array([r["r_max_mean_ratio"] for r in records], type=pa.float32()),
        "r_per_feature_summary":    list_col("r_per_feature_summary", pa.float32()),
        "r_signed_feature_summary": list_col("r_signed_feature_summary", pa.float32()),
        "r_token_fire_counts":      list_col("r_token_fire_counts", pa.int32()),
        "a_token_fire_counts":      list_col("a_token_fire_counts", pa.int32()),
        "b_token_fire_counts":      list_col("b_token_fire_counts", pa.int32()),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    log.info(f"Saved → {out_path}  ({len(records)} rows)")


def main() -> None:
    cfg = CFG
    log.info(f"Device: {cfg['device']}")

    clips = select_clips(cfg)
    video_dir = Path(cfg["video_dir"])

    records = []
    with DFAEngine(
        model_flag=cfg["model_flag"],
        sae_path=cfg["sae_path"],
        dim_mean_path=cfg["dim_mean_path"],
        layer=cfg["layer"],
        device=cfg["device"],
    ) as engine:
        for i, (clip_id, class_id, class_name) in enumerate(clips):
            log.info(f"[{i+1}/{len(clips)}] clip={clip_id}  class={class_id} ({class_name})")
            record = run_clip(engine, clip_id, class_id, class_name, video_dir, cfg)
            records.append(record)
            log.info(
                f"  R: correct={record['r_correct']}  logit={record['r_logit']:.2f}"
                f"  margin={record['r_logit_margin']:.2f}  mmr={record['r_max_mean_ratio']:.1f}"
            )

    save_parquet(records, Path(cfg["output_path"]))


if __name__ == "__main__":
    main()
