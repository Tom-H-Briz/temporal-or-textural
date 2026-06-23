"""
Single-class R vs C DFA analysis for TF SAE features.

For a given class ID: runs DFA under real video (R) and shuffled video (C)
on R-correct clips, then aggregates per-feature means and delta = mean_s_R − mean_s_C.

Usage:
    uv run python src/stage3_analysis/clip_analysis_tf.py 36
"""

import argparse
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(ROOT / "notebooks"))

from perturbation import apply_shuffle
from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

_LAYER = 7

CFG = {
    "model_flag":      "timesformer",
    "layer":           _LAYER,
    "labels_path":     str(ROOT / "data/ssv2/labels/labels.json"),
    "validation_path": str(ROOT / "data/ssv2/labels/validation.json"),
    "video_dir":       str(ROOT / "data/ssv2/20bn-something-something-v2"),
    "output_dir":      str(ROOT / "outputs/stage3_analysis/clip_analysis_tf"),
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
}


def _resolve_cfg(cfg: dict) -> dict:
    sae_dir   = ROOT / "outputs" / "sae"
    layer     = cfg["layer"]
    matches   = list(sae_dir.glob(f"sae_tf_k*_x*_l{layer}_job{layer}_best.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected 1 TF checkpoint for layer {layer}, found: {matches}")
    ckpt_path = matches[0]
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sae_k     = ckpt.get("sae_k")
    if sae_k is None:
        raise ValueError(f"sae_k not in checkpoint {ckpt_path}")
    dim_mean  = sae_dir / f"tf_layer{layer}_dim_mean.pt"
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(ckpt_path), "dim_mean_path": str(dim_mean), "sae_k": sae_k}


def load_clips(cfg: dict, class_id: int) -> list[tuple[str, Path]]:
    label_map, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    result    = []
    for c in clips:
        if label_map.get(_strip_brackets(c["template"])) != class_id:
            continue
        path = video_dir / f"{c['id']}.webm"
        if path.exists():
            result.append((str(c["id"]), path))
    print(f"  Class {class_id}: {len(result)} clips on disk")
    return result


def preprocess_c(clip_path: Path, clip_id: str, num_frames: int, processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, int(clip_id) % 2**32)
    n      = len(frames)
    idx    = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def collect_scores(engine: DFAEngine, clips: list[tuple[str, Path]],
                   class_id: int, cfg: dict) -> list[dict]:
    records = []
    for clip_id, clip_path in clips:
        r_result = engine.run(clip_path, class_id)
        if not r_result.correct:
            continue
        pv_c     = preprocess_c(clip_path, clip_id, engine._num_frames,
                                engine._processor, cfg["device"])
        c_result = engine.run_pixels(pv_c, class_id)
        records.append({
            "clip_id": clip_id,
            "dfa":     r_result.per_feature_summary.numpy(),
            "s_R":     r_result.signed_feature_summary.numpy(),
            "s_C":     c_result.signed_feature_summary.numpy(),
        })
    print(f"  Class {class_id}: {len(records)} R-correct clips used")
    return records


def aggregate(records: list[dict]) -> pd.DataFrame:
    mean_dfa = np.stack([r["dfa"] for r in records]).mean(axis=0)
    mean_s_R = np.stack([r["s_R"] for r in records]).mean(axis=0)
    mean_s_C = np.stack([r["s_C"] for r in records]).mean(axis=0)
    return pd.DataFrame({
        "feature_idx": np.arange(len(mean_dfa)),
        "mean_dfa":    mean_dfa,
        "mean_s_R":    mean_s_R,
        "mean_s_C":    mean_s_C,
        "delta":       mean_s_R - mean_s_C,
    })


def save_outputs(records: list[dict], df: pd.DataFrame, class_id: int, cfg: dict) -> None:
    out_dir  = Path(cfg["output_dir"])
    csv_name = f"clip_analysis_tf_{class_id}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked = df.sort_values("mean_dfa", ascending=False).reset_index(drop=True)
    cols   = ["feature_idx", "mean_dfa", "mean_s_R", "mean_s_C", "delta"]
    out    = ranked[cols].copy()
    num_cols  = [c for c in cols if c != "feature_idx"]
    total_row = {c: out[c].sum() for c in num_cols}
    total_row["feature_idx"] = "total"
    out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)
    out.to_csv(out_dir / csv_name, index=False)

    with open(out_dir / f"run_summary_{class_id}.txt", "w") as f:
        f.write(f"class {class_id}: {len(records)} R-correct clips used\n")
        f.write(f"features total: {len(df)}\n")
        f.write(f"mean_dfa range: [{ranked['mean_dfa'].min():.4f}, {ranked['mean_dfa'].max():.4f}]\n")
        top10 = ranked.head(10)[["feature_idx", "mean_dfa", "delta"]].values.tolist()
        f.write(f"top-10 by mean_dfa: {top10}\n")
    print(f"  Outputs → {out_dir / csv_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-class R vs C DFA analysis")
    parser.add_argument("class_id", type=int, help="SSv2 class ID to analyse")
    args = parser.parse_args()

    cfg = {**CFG, **_resolve_cfg(CFG)}
    print(f"Device: {cfg['device']}  Layer: {cfg['layer']}  Class: {args.class_id}")
    print(f"SAE: {Path(cfg['sae_path']).name}  sae_k={cfg['sae_k']}")

    clips = load_clips(cfg, args.class_id)

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=cfg["sae_k"]) as engine:
        records = collect_scores(engine, clips, args.class_id, cfg)

    df = aggregate(records)
    save_outputs(records, df, args.class_id, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
