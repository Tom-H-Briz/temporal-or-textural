"""
Max-activating clip search + render — TF heavy dozen, both datasets.

K400: top-12 by mass (03/10 findings, signflip_ablation_candidates.csv).
SSv2: the registered TOP12 (run_ablation_tf.py; validated as top-12 by mass).

Forward pass only (get_z), no DFA. Scans every SL-manifest clip once and
computes stats for all requested features from the same z. Renders each
feature's top-k clips: sampled frames with per-frame spatial activation
overlays, titled with feature, rank, clip_id, class name and SL label.

TF token layout: patch-major/frame-minor — token_idx = patch*8 + frame, so
frame = token_idx % 8, patch = token_idx // 8 (gather_by_position convention).

Usage:
    uv run python src/stage3_analysis/visualisation/max_activating_clips_tf.py --dataset kinetics400
    uv run python src/stage3_analysis/visualisation/max_activating_clips_tf.py --dataset ssv2
    ... --features 3711 4749        # subset
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from ToT_utils import (
    MODEL_REGISTRY, _strip_brackets, get_frame_sampler, load_clips_kinetics,
    load_metadata, resolve_k400_label2id, resolve_sae_checkpoint,
)
from stage3_analysis.dfa_engine import DFAEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Heavy dozens — k64/x8 dictionary indices, layer-specific, per dataset.
DEFAULT_FEATURES = {
    "kinetics400": [3711, 4749, 860, 4924, 1541, 3500, 6018, 3757, 5198, 719, 3206, 1486],
    "ssv2":        [3029, 1517, 2090, 2057, 2156, 1588, 4590, 3813, 6029, 622, 1371, 4134],  # TOP12
}

CFG = {
    "model_flag":     "timesformer",
    "layer":          int(os.environ.get("SAE_LAYER", 7)),
    "device":         "cuda" if torch.cuda.is_available() else "cpu",
    "top_k":          5,
    "out_dir":        str(ROOT / "outputs/analysis/max_activating_tf"),
    "kinetics400": {
        "video_dir":     os.environ.get("VIDEO_DIR", str(ROOT / "data/kinetics400/val")),
        "k400_manifest": str(ROOT / "outputs/Laura_SL/k400_manifest_SL_subset.json"),
        "k400_sl_csv":   str(ROOT / "outputs/Laura_SL/k400_sl_class_mapping.csv"),
    },
    "ssv2": {
        "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
        "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
        "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
        "manifest_path":   str(ROOT / "outputs/Laura_SL/manifest_SL_subset.json"),
        "sl_csv":          str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    },
}

_NF = MODEL_REGISTRY["timesformer"]["num_frames"]      # 8


def load_clips_ssv2(cfg: dict) -> list[tuple[str, int, Path]]:
    """SSv2 SL-manifest clips (all of them — activation is not correctness-gated),
    mirroring dfa_mass_delta.py's loader. Returns (clip_id, class_id, path)."""
    import json
    label_map, _, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)
    result = []
    for entries in manifest.values():
        for entry in entries:
            cid = label_map.get(_strip_brackets(entry["template"]))
            path = video_dir / f"{entry['id']}.webm"
            if cid is not None and path.exists():
                result.append((str(entry["id"]), cid, path))
    log.info(f"  {len(result):,} clips from SSv2 SL manifest")
    return result


def scan(engine: DFAEngine, clips: list, features: list[int], sampler) -> dict[int, list[dict]]:
    """One pass over clips; per feature: total/peak activation and peak frame."""
    records = {f: [] for f in features}
    for i, (clip_id, class_id, clip_path) in enumerate(clips):
        try:
            z = engine.get_z(clip_path, frame_sampler=sampler)   # (1568, dict)
        except Exception as exc:
            log.warning(f"SKIP {clip_id}: {exc}")
            continue
        for f in features:
            act = z[:, f]
            tok = int(act.argmax().item())
            records[f].append({
                "clip_id": clip_id, "class_id": class_id, "clip_path": str(clip_path),
                "total_activation": act.sum().item(), "peak_activation": act.max().item(),
                "peak_frame": tok % _NF,       # patch-major/frame-minor layout
                "peak_patch": tok // _NF,      # 0..195 — spatial argmax, for the
                                               # within-class positional-alignment angle
            })
        if (i + 1) % 500 == 0:
            log.info(f"  [{i+1}/{len(clips)}] scanned")
    return records


def decode_sampled(clip_path: Path, sampler) -> np.ndarray:
    """Decode a clip and return the 8 sampler-selected RGB frames (H,W,3)."""
    with av.open(str(clip_path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    idx = sampler(len(frames), _NF)
    return np.stack([frames[i] for i in idx])


def frame_maps(z: torch.Tensor, feat_idx: int) -> np.ndarray:
    """Feature activation reshaped to (8 frames, 14, 14) via the TF layout.
    .cpu() first — get_z returns a CUDA tensor and numpy can't touch it."""
    per_patch = z[:, feat_idx].detach().cpu().numpy().reshape(196, _NF).T      # (8, 196)
    return per_patch.reshape(_NF, 14, 14)


def render_clip(engine: DFAEngine, row: pd.Series, feat_idx: int, class_name: str,
                sl_label: str, sampler, out_dir: Path) -> None:
    """Grid: top row sampled frames, bottom row frames + 14x14 activation overlay."""
    rgb = decode_sampled(Path(row["clip_path"]), sampler)
    z = engine.get_z(Path(row["clip_path"]), frame_sampler=sampler)
    maps = frame_maps(z, feat_idx)

    fig, axes = plt.subplots(2, _NF, figsize=(2.2 * _NF, 5.4))
    for t in range(_NF):
        axes[0, t].imshow(rgb[t]); axes[0, t].set_title(f"t{t}", fontsize=8)
        axes[1, t].imshow(rgb[t])
        heat = np.kron(maps[t], np.ones((16, 16)))[:rgb[t].shape[0], :rgb[t].shape[1]]
        axes[1, t].imshow(heat, cmap="jet", alpha=0.45,
                          extent=(0, rgb[t].shape[1], rgb[t].shape[0], 0))
        axes[1, t].set_title(f"act {maps[t].mean():.2f}", fontsize=8)
        for ax in (axes[0, t], axes[1, t]):
            ax.axis("off")
    fig.suptitle(f"feature {feat_idx}  rank {int(row['rank'])}  clip {row['clip_id']}  "
                 f"class {int(row['class_id'])}: {class_name}  [{sl_label}]  "
                 f"total_act {row['total_activation']:.1f}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    safe = class_name.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:30]
    fig.savefig(out_dir / f"f{feat_idx}_rank{int(row['rank']):02d}_"
                          f"clip{row['clip_id']}_c{int(row['class_id'])}_{safe}.png", dpi=110)
    plt.close(fig)


def process_feature(feat_idx: int, records: list[dict], id2name: dict, sl_map: dict,
                    engine: DFAEngine, sampler, out_dir: Path) -> None:
    df = pd.DataFrame(records).sort_values("total_activation", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    top = df.head(CFG["top_k"]).copy()
    top["class_name"] = top["class_id"].map(id2name)
    top["sl_label"] = top["class_id"].map(sl_map).fillna("unlabelled")
    fdir = out_dir / f"feature{feat_idx}"; fdir.mkdir(parents=True, exist_ok=True)
    # Full scan table (all clips) — the raw material for within-class positional
    # alignment; top.csv is just the render list.
    df.to_csv(fdir / "scan.csv", index=False)
    top[["rank", "clip_id", "class_id", "class_name", "sl_label",
         "total_activation", "peak_activation", "peak_frame", "peak_patch"]].to_csv(
             fdir / "top.csv", index=False)
    log.info(f"feature {feat_idx}: top5 classes -> {top['class_name'].tolist()}")
    for _, row in top.iterrows():
        # One bad clip must not kill the remaining renders (same lesson as the
        # train_sae epilogue): log, skip, keep going.
        try:
            render_clip(engine, row, feat_idx, row["class_name"], row["sl_label"], sampler, fdir)
        except Exception as exc:
            log.warning(f"RENDER SKIP feature {feat_idx} rank {int(row['rank'])} "
                        f"clip {row['clip_id']}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kinetics400", "ssv2"], default="kinetics400")
    parser.add_argument("--features", type=int, nargs="+", default=None,
                        help="default: the dataset's heavy dozen")
    args = parser.parse_args()
    features = args.features if args.features else DEFAULT_FEATURES[args.dataset]
    layer = CFG["layer"]
    resolved = resolve_sae_checkpoint("timesformer", layer, dataset_name=args.dataset,
                                      sae_k=64, job_label="7ep")
    sampler = get_frame_sampler(args.dataset, MODEL_REGISTRY["timesformer"])
    dcfg = CFG[args.dataset]
    if args.dataset == "kinetics400":
        clips = load_clips_kinetics(dcfg["k400_manifest"], dcfg["video_dir"], "timesformer")
        id2name = {v: k for k, v in resolve_k400_label2id("timesformer").items()}
        sl_df = pd.read_csv(dcfg["k400_sl_csv"]).dropna(subset=["matched_model_class_id"])
        sl_map = {int(r["matched_model_class_id"]): r["sl_category"] for _, r in sl_df.iterrows()}
    else:
        clips = load_clips_ssv2(dcfg)
        _, _, id2name = load_metadata(dcfg["labels_path"], dcfg["validation_path"])
        sl_map = {int(r["class_id"]): r["category"]
                  for _, r in pd.read_csv(dcfg["sl_csv"]).iterrows()}
    log.info(f"{len(clips):,} clips | {args.dataset} | features {features} | layer {layer}")

    out_dir = Path(CFG["out_dir"]) / f"{args.dataset}_l{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with DFAEngine("timesformer", resolved["sae_path"], resolved["dim_mean_path"],
                   layer=layer, device=CFG["device"], sae_k=resolved["sae_k"],
                   dataset_name=args.dataset) as engine:
        records = scan(engine, clips, features, sampler)
        for f in features:
            process_feature(f, records[f], id2name, sl_map, engine, sampler, out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
