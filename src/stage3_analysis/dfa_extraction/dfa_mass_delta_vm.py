"""
DFA mass delta diagnostic (VideoMAE or TimeSformer) — R vs shuffle vs A across
SL manifest clips. Shuffle condition is model-dispatched: VM = C1 (shuffled
consecutive tubelet pairs), TF = C (raw pre-sampling frame shuffle).

C1/C seed = _deterministic_seed(clip_id) (int for SSv2, crc32 for K400 ids).
For each R-correct clip: delta = sum(abs(DFA_R)) - sum(abs(DFA_shuffle))

signed_vec_R/C1/A are checkpoint-width vectors (dict_size varies by SAE config,
e.g. 6144 at 8x expansion vs 12288 at 16x) — feature indices are only meaningful
within the SAE that produced them, never comparable across configs.

Outputs:
    VM:  outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_{dataset}_l{layer}_job{job}_k{k}.parquet
    TF:  outputs/analysis/dfa_mass_delta_tf/dfa_mass_delta_tf_{dataset}_l{layer}_job{job}_k{k}.parquet

Usage:
    uv run python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py --layer 7
    uv run python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py --layer 7 --sae-k 128
    uv run python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py --dataset kinetics400 --layer 7
    uv run python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py --model timesformer --dataset kinetics400 --layer 7
"""

import argparse
import os
import sys
from pathlib import Path

import av
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(ROOT / "notebooks"))

from perturbation import apply_shuffle
from perturbationA import apply_midpoint_frame
from ToT_utils import (
    MODEL_REGISTRY, _deterministic_seed, _strip_brackets, get_frame_sampler,
    load_metadata, resolve_sae_checkpoint,
)
from ToT_utils import load_clips_kinetics as tot_load_clips_kinetics
from stage3_analysis.dfa_engine import DFAEngine

SL_COLOURS = {"temporal": "steelblue", "static": "darkorange"}

CFG = {
    "model_flag":      "videomae",
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "dfa_classes":     [0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40, 41,
                        42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164, 168, 169, 171, 173],
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "manifest_path":   str(ROOT / "outputs/Laura_SL/manifest_SL_subset.json"),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "k400_manifest_path": str(ROOT / "outputs/Laura_SL/k400_manifest_SL_subset.json"),
    "k400_sl_csv_path": str(ROOT / "outputs/Laura_SL/k400_sl_class_mapping.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/dfa_mass_delta_vm_c1"),
}


def build_sl_label_map(cfg: dict, dataset_name: str) -> dict[int, str]:
    if dataset_name == "kinetics400":
        df = pd.read_csv(cfg["k400_sl_csv_path"]).dropna(subset=["matched_model_class_id"])
        return {int(row["matched_model_class_id"]): row["sl_category"] for _, row in df.iterrows()}
    df = pd.read_csv(cfg["sl_csv_path"])
    return {int(row["class_id"]): row["category"] for _, row in df.iterrows()}


def load_clips_ssv2(cfg: dict) -> list[tuple[str, int, Path]]:
    label_map, _, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)
    result = []
    for entries in manifest.values():
        for entry in entries:
            cid   = label_map.get(_strip_brackets(entry["template"]))
            path_r = video_dir / f"{entry['id']}.webm"
            if cid is not None and path_r.exists():
                result.append((str(entry["id"]), cid, path_r))
    print(f"  {len(result)} clips from SL manifest")
    return result


def load_clips_kinetics(cfg: dict) -> list[tuple[str, int, Path]]:
    """Thin cfg-unpacking wrapper — see ToT_utils.load_clips_kinetics for the
    shared logic (manifest schema, why label2id is resolved at call time, why
    no held-out/correctness filtering) and position_lock_extraction.py for the
    other caller this was consolidated from (03/08)."""
    result = tot_load_clips_kinetics(cfg["k400_manifest_path"], cfg["video_dir"], cfg["model_flag"])
    print(f"  {len(result)} clips from K400 SL manifest")
    return result


def load_clips(cfg: dict, dataset_name: str) -> list[tuple[str, int, Path]]:
    if dataset_name == "kinetics400":
        return load_clips_kinetics(cfg)
    return load_clips_ssv2(cfg)


def preprocess_c1(clip_path: Path, clip_id: str, num_frames: int, processor, device: str,
                  frame_sampler) -> torch.Tensor:
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


def preprocess_c(clip_path: Path, clip_id: str, num_frames: int, processor, device: str,
                 frame_sampler) -> torch.Tensor:
    """TF's raw-frame shuffle — ALL native frames permuted pre-sampling (matches
    perturb_accuracy_tf.py / dfa_mass_delta.py's preprocess_c). In-grammar for
    TF's single-frame tokens; VM uses C1 instead (tubelet-preserving pairs)."""
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, _deterministic_seed(clip_id))
    n      = len(frames)
    idx    = frame_sampler(n, num_frames)
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def preprocess_a(clip_path: Path, num_frames: int, processor, device: str,
                 frame_sampler) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_midpoint_frame(frames)
    n      = len(frames)
    idx    = frame_sampler(n, num_frames)
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def run_clips(engine: DFAEngine, clips: list[tuple[str, int, Path]], cfg: dict,
              frame_sampler) -> list[dict]:
    # Shuffle condition is model-dispatched (position_lock_extraction.py precedent):
    # VM = C1 (tubelet pairs), TF = C (raw pre-sampling shuffle).
    shuffle_fn = preprocess_c1 if cfg["model_flag"] == "videomae" else preprocess_c
    shuf_label = "C1" if cfg["model_flag"] == "videomae" else "C"  # parquet column suffix
    records = []
    for i, (clip_id, class_id, path_r) in enumerate(clips):
        r_result = engine.run(path_r, class_id, frame_sampler=frame_sampler)
        if not r_result.correct:
            continue
        pv_shuf  = shuffle_fn(path_r, clip_id, engine._num_frames, engine._processor,
                              cfg["device"], frame_sampler)
        shuf_result = engine.run_pixels(pv_shuf, class_id)
        # skip_a (--skip-a): A columns kept in the schema as nulls, so readers are unchanged.
        a_result = None if cfg.get("skip_a") else engine.run_pixels(
            preprocess_a(path_r, engine._num_frames, engine._processor, cfg["device"], frame_sampler), class_id)
        s_r   = r_result.signed_feature_summary.numpy().astype(np.float32)
        s_shu = shuf_result.signed_feature_summary.numpy().astype(np.float32)
        s_a   = None if a_result is None else a_result.signed_feature_summary.numpy().astype(np.float32)
        records.append({
            "clip_id":         clip_id,
            "class_id":        class_id,
            "total_abs_R":     float(r_result.per_feature_summary.sum()),
            f"total_abs_{shuf_label}":    float(shuf_result.per_feature_summary.sum()),
            "total_abs_A":     np.nan if a_result is None else float(a_result.per_feature_summary.sum()),
            "delta":           float(r_result.per_feature_summary.sum() - shuf_result.per_feature_summary.sum()),
            f"correct_{shuf_label}":      bool(shuf_result.correct),
            "correct_A":       None if a_result is None else bool(a_result.correct),
            "total_signed_R":  float(s_r.sum()),
            f"total_signed_{shuf_label}": float(s_shu.sum()),
            "total_signed_A":  np.nan if s_a is None else float(s_a.sum()),
            "signed_vec_R":    s_r,
            f"signed_vec_{shuf_label}":   s_shu,
            "signed_vec_A":    s_a,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(clips)}] R-correct so far: {len(records)}")
    print(f"  Done: {len(records)} R-correct clips from {len(clips)} total")
    return records


def save_parquet(records: list[dict], sl_map: dict[int, str], out_dir: Path, out_suffix: str) -> None:
    df = pd.DataFrame(records)
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
def save_parquet(records: list[dict], sl_map: dict[int, str], out_dir: Path,
                 out_prefix: str, shuf_label: str, out_suffix: str) -> None:
    df = pd.DataFrame(records)
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    # Column order mirrors the historical VM layout with the shuffle label inlined —
    # VM output is byte-compatible (C1), TF writes the same shape with C.
    df = df[["clip_id", "class_id", "sl_label",
             "total_abs_R", f"total_abs_{shuf_label}", "total_abs_A", "delta",
             f"correct_{shuf_label}", "correct_A",
             "total_signed_R", f"total_signed_{shuf_label}", "total_signed_A",
             "signed_vec_R", f"signed_vec_{shuf_label}", "signed_vec_A"]]
    path = out_dir / f"{out_prefix}_{out_suffix}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Parquet → {path}  ({len(df)} rows)")


def make_plot(records: list[dict], sl_map: dict[int, str], out_dir: Path,
              shuf_label: str, out_suffix: str) -> None:
    df = pd.DataFrame(records)
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    df = df.sort_values("delta").reset_index(drop=True)
    df["y"] = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10, 7))
    for label, colour in SL_COLOURS.items():
        grp = df[df["sl_label"] == label]
        ax.scatter(grp["delta"], grp["y"], s=8, c=colour, alpha=0.6,
                   label=f"{label.capitalize()} (n={len(grp)})")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"delta  (total_abs_R − total_abs_{shuf_label})")
    ax.set_ylabel("clip rank (sorted by delta ascending)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"dfa_mass_delta_{out_suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["videomae", "timesformer", "umt"], default="videomae")
    parser.add_argument("--dataset", choices=["ssv2", "kinetics400"], default="ssv2")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--job-label", type=str, default="7ep")
    parser.add_argument("--sae-k", type=int, default=64, help="fallback if checkpoint lacks sae_k")
    parser.add_argument("--manifest", type=str, default=None,
                        help="ssv2 clip manifest (same schema as manifest_SL_subset.json); requires --out-tag")
    parser.add_argument("--out-tag", type=str, default=None, help="filename tag, e.g. all174 — keeps SL outputs intact")
    parser.add_argument("--skip-a", action="store_true", help="R and shuffle only; A columns written as nulls")
    args = parser.parse_args()
    assert args.manifest is None or args.out_tag, "--manifest needs --out-tag (never overwrite the SL parquets)"

    resolved   = resolve_sae_checkpoint(args.model, args.layer, dataset_name=args.dataset,
                                        sae_k=args.sae_k, job_label=args.job_label)
    cfg           = {**CFG, **resolved, "model_flag": args.model, "layer": args.layer, "skip_a": args.skip_a}
    if args.manifest:
        cfg["manifest_path"] = args.manifest
    frame_sampler = get_frame_sampler(args.dataset, MODEL_REGISTRY[args.model])
    shuf_label    = "C1" if args.model == "videomae" else "C"
    # UMT gets its own prefix/dir: same C as TF (single-frame tokens) but must not land in TF's files.
    out_prefix    = {"videomae": "dfa_mass_delta_vm_c1", "timesformer": "dfa_mass_delta_tf",
                     "umt": "dfa_mass_delta_umt"}[args.model]
    tag           = f"_{args.out_tag}" if args.out_tag else ""
    out_suffix    = f"{args.dataset}{tag}_l{args.layer}_job{resolved['job_label']}_k{resolved['sae_k']}"
    print(f"Device: {cfg['device']}  Layer: {cfg['layer']}  Dataset: {args.dataset}  Model: {args.model}")
    print(f"SAE: {Path(cfg['sae_path']).name}  sae_k={cfg['sae_k']}  shuffle condition: {shuf_label}")

    out_dir = Path(cfg["output_dir"] if args.model == "videomae"
                   else str(ROOT / "outputs" / "analysis" / out_prefix))
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = build_sl_label_map(cfg, args.dataset)
    clips  = load_clips(cfg, args.dataset)

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=cfg["sae_k"], dataset_name=args.dataset) as engine:
        records = run_clips(engine, clips, cfg, frame_sampler)

    save_parquet(records, sl_map, out_dir, out_prefix, shuf_label, out_suffix)
    make_plot(records, sl_map, out_dir, shuf_label, out_suffix)
    print("Done.")


if __name__ == "__main__":
    main()
