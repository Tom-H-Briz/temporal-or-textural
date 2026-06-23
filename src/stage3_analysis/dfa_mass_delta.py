"""
DFA mass delta diagnostic — R vs C total causal mass across 32 SL classes.

For each R-correct clip: delta = sum(abs(DFA_R)) - sum(abs(DFA_C))
Positive: R has more causal mass (temporal signal attenuated under shuffle)
Negative: C recruited more mass (motion-noise over-recruitment)

Outputs:
    outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet
    outputs/analysis/dfa_mass_delta/dfa_mass_delta.png
"""

import os
import sys
from pathlib import Path

import av
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
from ToT_utils import _strip_brackets, load_metadata
from stage3_analysis.dfa_engine import DFAEngine

SL_COLOURS = {"temporal": "steelblue", "static": "darkorange"}

CFG = {
    "model_flag":      "timesformer",
    "layer":           7,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "dfa_classes":     [0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40, 41,
                        42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164, 168, 169, 171, 173],
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "sl_csv_path":     str(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"),
    "output_dir":      str(ROOT / "outputs/analysis/dfa_mass_delta"),
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


def build_sl_label_map(cfg: dict) -> dict[int, str]:
    df = pd.read_csv(cfg["sl_csv_path"])
    return {int(row["class_id"]): row["category"] for _, row in df.iterrows()}


def load_clips(cfg: dict) -> list[tuple[str, int, Path]]:
    label_map, clips, _ = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    target    = set(cfg["dfa_classes"])
    result    = []
    for c in clips:
        cid = label_map.get(_strip_brackets(c["template"]))
        if cid not in target:
            continue
        path = video_dir / f"{c['id']}.webm"
        if path.exists():
            result.append((str(c["id"]), cid, path))
    print(f"  {len(result)} clips across {len(target)} DFA classes")
    return result


def preprocess_c(clip_path: Path, clip_id: str, num_frames: int, processor, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    frames = apply_shuffle(frames, int(clip_id) % 2**32)
    n      = len(frames)
    idx    = torch.linspace(0, n - 1, num_frames).long().tolist()
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def run_clips(engine: DFAEngine, clips: list[tuple[str, int, Path]], cfg: dict) -> list[dict]:
    records = []
    for i, (clip_id, class_id, clip_path) in enumerate(clips):
        r_result = engine.run(clip_path, class_id)
        if not r_result.correct:
            continue
        pv_c     = preprocess_c(clip_path, clip_id, engine._num_frames,
                                engine._processor, cfg["device"])
        c_result = engine.run_pixels(pv_c, class_id)
        s_r = r_result.signed_feature_summary.numpy().astype(np.float32)
        s_c = c_result.signed_feature_summary.numpy().astype(np.float32)
        records.append({
            "clip_id":        clip_id,
            "class_id":       class_id,
            "total_abs_R":    float(r_result.per_feature_summary.sum()),
            "total_abs_C":    float(c_result.per_feature_summary.sum()),
            "delta":          float(r_result.per_feature_summary.sum() - c_result.per_feature_summary.sum()),
            "correct_C":      bool(c_result.correct),
            "total_signed_R": float(s_r.sum()),
            "total_signed_C": float(s_c.sum()),
            "signed_vec_R":   s_r,
            "signed_vec_C":   s_c,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(clips)}] R-correct so far: {len(records)}")
    print(f"  Done: {len(records)} R-correct clips from {len(clips)} total")
    return records


def save_parquet(records: list[dict], sl_map: dict[int, str], out_dir: Path) -> None:
    df = pd.DataFrame(records)
    df["sl_label"] = df["class_id"].map(sl_map).fillna("unlabelled")
    df = df[["clip_id", "class_id", "sl_label",
             "total_abs_R", "total_abs_C", "delta", "correct_C",
             "total_signed_R", "total_signed_C", "signed_vec_R", "signed_vec_C"]]
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
