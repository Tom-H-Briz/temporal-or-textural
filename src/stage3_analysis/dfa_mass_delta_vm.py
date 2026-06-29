"""
DFA mass delta diagnostic (VideoMAE) — R vs C vs A across SL manifest clips.

For each R-correct clip: delta = sum(abs(DFA_R)) - sum(abs(DFA_C))

Outputs:
    outputs/analysis/dfa_mass_delta_vm/dfa_mass_delta.parquet
    outputs/analysis/dfa_mass_delta_vm/dfa_mass_delta.png
"""

import os
import sys
from pathlib import Path

import av
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(ROOT / "notebooks"))

from perturbation import apply_shuffle
from perturbationA import apply_midpoint_frame
from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

SL_COLOURS = {"temporal": "steelblue", "static": "darkorange"}

CFG = {
    "model_flag":      "videomae",
    "layer":           7,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "dfa_classes":     [0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40, 41,
                        42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164, 168, 169, 171, 173],
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "manifest_path":   str(ROOT / "outputs/Laura_SL/manifest_SL_subset.json"),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/dfa_mass_delta_vm"),
}


def _resolve_cfg(cfg: dict) -> dict:
    sae_path = ROOT / "outputs" / "sae" / "sae_layer7_job64.pt"
    dim_mean = ROOT / "outputs" / "sae" / "layer7_dim_mean.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"VM SAE checkpoint not found: {sae_path}")
    if not dim_mean.exists():
        raise FileNotFoundError(f"dim_mean not found: {dim_mean}")
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": 64}


def build_sl_label_map(cfg: dict) -> dict[int, str]:
    df = pd.read_csv(cfg["sl_csv_path"])
    return {int(row["class_id"]): row["category"] for _, row in df.iterrows()}


def load_clips(cfg: dict) -> list[tuple[str, int, Path]]:
    label_map, _, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    with open(cfg["manifest_path"]) as f:
        manifest = json.load(f)
    result = []
    for sl_label, entries in manifest.items():
        for entry in entries:
            cid   = label_map.get(_strip_brackets(entry["template"]))
            path_r = video_dir / f"{entry['id']}.webm"
            if cid is not None and path_r.exists():
                result.append((str(entry["id"]), cid, path_r))
    print(f"  {len(result)} clips from SL manifest")
    return result


def preprocess_c(clip_path: Path, clip_id: str, num_frames: int, processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, int(clip_id) % 2**32)
    n      = len(frames)
    idx    = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def preprocess_a(clip_path: Path, num_frames: int, processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_midpoint_frame(frames)
    n      = len(frames)
    idx    = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def run_clips(engine: DFAEngine, clips: list[tuple[str, int, Path]], cfg: dict) -> list[dict]:
    records = []
    for i, (clip_id, class_id, path_r) in enumerate(clips):
        r_result = engine.run(path_r, class_id)
        if not r_result.correct:
            continue
        pv_c     = preprocess_c(path_r, clip_id, engine._num_frames, engine._processor, cfg["device"])
        pv_a     = preprocess_a(path_r, engine._num_frames, engine._processor, cfg["device"])
        c_result = engine.run_pixels(pv_c, class_id)
        a_result = engine.run_pixels(pv_a, class_id)
        s_r = r_result.signed_feature_summary.numpy().astype(np.float32)
        s_c = c_result.signed_feature_summary.numpy().astype(np.float32)
        s_a = a_result.signed_feature_summary.numpy().astype(np.float32)
        records.append({
            "clip_id":        clip_id,
            "class_id":       class_id,
            "total_abs_R":    float(r_result.per_feature_summary.sum()),
            "total_abs_C":    float(c_result.per_feature_summary.sum()),
            "total_abs_A":    float(a_result.per_feature_summary.sum()),
            "delta":          float(r_result.per_feature_summary.sum() - c_result.per_feature_summary.sum()),
            "correct_C":      bool(c_result.correct),
            "correct_A":      bool(a_result.correct),
            "total_signed_R": float(s_r.sum()),
            "total_signed_C": float(s_c.sum()),
            "total_signed_A": float(s_a.sum()),
            "signed_vec_R":   s_r,
            "signed_vec_C":   s_c,
            "signed_vec_A":   s_a,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(clips)}] R-correct so far: {len(records)}")
    print(f"  Done: {len(records)} R-correct clips from {len(clips)} total")
    return records


def save_parquet(records: list[dict], sl_map: dict[int, str], out_dir: Path) -> None:
    df = pd.DataFrame(records)
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    df = df[["clip_id", "class_id", "sl_label",
             "total_abs_R", "total_abs_C", "total_abs_A", "delta", "correct_C", "correct_A",
             "total_signed_R", "total_signed_C", "total_signed_A",
             "signed_vec_R", "signed_vec_C", "signed_vec_A"]]
    path = out_dir / "dfa_mass_delta.parquet"
    df.to_parquet(path, index=False)
    print(f"  Parquet → {path}  ({len(df)} rows)")


def make_plot(records: list[dict], sl_map: dict[int, str], out_dir: Path) -> None:
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
    ax.set_xlabel("delta  (total_abs_R − total_abs_C)")
    ax.set_ylabel("clip rank (sorted by delta ascending)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "dfa_mass_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot → {path}")


def main() -> None:
    cfg = {**CFG, **_resolve_cfg(CFG)}
    print(f"Device: {cfg['device']}  Layer: {cfg['layer']}")
    print(f"SAE: {Path(cfg['sae_path']).name}  sae_k={cfg['sae_k']}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sl_map = build_sl_label_map(cfg)
    clips  = load_clips(cfg)

    with DFAEngine(cfg["model_flag"], cfg["sae_path"], cfg["dim_mean_path"],
                   layer=cfg["layer"], device=cfg["device"],
                   sae_k=cfg["sae_k"]) as engine:
        records = run_clips(engine, clips, cfg)

    save_parquet(records, sl_map, out_dir)
    make_plot(records, sl_map, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
