"""
Max-activating clips search — forward pass only, no DFA.

Scans all 32 SL-class clips once, computing activations for all requested
features from the same z — one model pass per clip regardless of feature count.

Usage (single feature):
    uv run python src/stage3_analysis/max_activating_clips.py

Usage (batch — multiple features):
    uv run python src/stage3_analysis/max_activating_clips.py --features 990 2705 1996 308

Token ordering: temporal-major — confirmed at feature_vis_group_vm.py:176.
peak_tubelet = peak_token_idx // 196
"""

import argparse
import logging
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
import stage3_analysis.feature_vis_group_vm as vis

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

CFG = {
    "model_flag":      "videomae",
    "layer":           7,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "feature_idx":     1990,          # default — overridden by --features
    "top_k":           10,
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
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


def scan(engine: DFAEngine, clips: list, features: list[int]) -> dict[int, list[dict]]:
    """Single pass over clips — compute stats for all features from each z."""
    records = {f: [] for f in features}
    for i, (clip_id, class_id, clip_path) in enumerate(clips):
        try:
            z = engine.get_z(clip_path)       # (1568, dict_size)
        except Exception as exc:
            log.warning(f"SKIP {clip_id}: {exc}")
            continue

        for feat_idx in features:
            feat         = z[:, feat_idx]     # (1568,)
            total_act    = feat.sum().item()
            peak_act     = feat.max().item()
            peak_tok     = int(feat.argmax().item())
            records[feat_idx].append({
                "clip_id":          clip_id,
                "class_id":         class_id,
                "clip_path":        str(clip_path),
                "total_activation": total_act,
                "peak_activation":  peak_act,
                "peak_tubelet":     peak_tok // 196,
            })

        if (i + 1) % 500 == 0:
            log.info(f"  [{i+1}/{len(clips)}] scanned")

    return records


def render_vis(top: pd.DataFrame, feat_idx: int, cfg: dict, resolved: dict,
               model, processor, sae, dim_mean, out_dir: Path) -> None:
    vis_cfg = {
        "model_flag":   cfg["model_flag"],
        "layer":        cfg["layer"],
        "num_frames":   16,
        "num_tubelets": 8,
        "n_spatial":    196,
    }
    W_dec = vis.get_decoder_weights(sae)
    for _, row in top.iterrows():
        clip_path = Path(row["clip_path"])
        clip_id   = row["clip_id"]
        frames    = vis.load_video_frames(clip_id, clip_path.parent, vis_cfg["num_frames"])
        z         = vis.extract_activations(frames, model, processor, sae, dim_mean,
                                            vis_cfg, cfg["device"])
        acts      = vis.signed_activation_map(z, feat_idx, W_dec,
                                               vis_cfg["num_tubelets"], vis_cfg["n_spatial"])
        vis.make_feature_image(
            clips_activations=[acts],
            clips_frames=[frames[::2]],
            clip_ids=[clip_id],
            feature_idx=feat_idx,
            class_id=int(row["class_id"]),
            condition="R",
            output_path=out_dir / f"rank{int(row['rank']):02d}_clip{clip_id}_class{int(row['class_id'])}.png",
            num_cols=vis_cfg["num_tubelets"],
        )


def process_feature(feat_idx: int, records: list[dict], sl_map: dict,
                    cfg: dict, resolved: dict, model, processor, sae, dim_mean) -> None:
    out_dir = ROOT / f"outputs/analysis/max_activating_clips_{feat_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    df = df.sort_values("total_activation", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    top = df.head(cfg["top_k"])

    csv_path = out_dir / f"max_activating_clips_{feat_idx}.csv"
    top[["rank", "clip_path", "class_id", "sl_label",
         "total_activation", "peak_activation", "peak_tubelet"]].to_csv(csv_path, index=False)
    log.info(f"Feature {feat_idx} → {csv_path}")
    print(top[["rank", "clip_id", "class_id", "sl_label",
               "total_activation", "peak_tubelet"]].to_string(index=False))

    render_vis(top, feat_idx, cfg, resolved, model, processor, sae, dim_mean, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, nargs="+",
                        help="Feature indices to search (default: CFG['feature_idx'])")
    args = parser.parse_args()

    features = args.features if args.features else [CFG["feature_idx"]]
    cfg      = CFG
    resolved = _resolve_cfg()

    log.info(f"Device: {cfg['device']}  Features: {features}")
    sl_map = {int(r["class_id"]): r["category"]
              for _, r in pd.read_csv(cfg["sl_csv_path"]).iterrows()}
    clips  = load_clips(cfg)

    log.info(f"Scanning {len(clips):,} clips for {len(features)} feature(s)…")
    with DFAEngine(cfg["model_flag"], resolved["sae_path"], resolved["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=resolved["sae_k"]) as engine:
        all_records = scan(engine, clips, features)

    log.info("Loading vis model for heatmap rendering…")
    vis_cfg = {"model_flag": cfg["model_flag"], "layer": cfg["layer"],
               "num_frames": 16, "num_tubelets": 8, "n_spatial": 196}
    model, processor, sae, dim_mean = vis.load_model_and_sae(vis_cfg, resolved, cfg["device"])

    for feat_idx in features:
        log.info(f"Processing feature {feat_idx}…")
        process_feature(feat_idx, all_records[feat_idx], sl_map,
                        cfg, resolved, model, processor, sae, dim_mean)

    log.info("Done.")


if __name__ == "__main__":
    main()
