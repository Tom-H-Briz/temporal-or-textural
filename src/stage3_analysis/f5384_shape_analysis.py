"""
f5384_shape_analysis.py

Stage 1: extract per-tubelet activation stats for feature 5384 across
         class 18 (main) and class 0 (control/hum baseline).
Stage 2: classify temporal shape, join DFA sign, compute contingency table,
         onset/offset distributions, and heatmap.

Outputs (outputs/analysis/f5384_shape/):
    f5384_raw.parquet            — 8 rows per clip, Stage 1 floor
    f5384_shape.csv              — per-clip shape + sign + threshold agreement
    f5384_none_clips.csv         — clips where shape == "none" (for inspection)
    f5384_contingency.csv        — shape × dfa_sign, counts + row %
    f5384_onset_offset.csv       — per-shape onset/offset distributions
    f5384_heatmap.png            — class-18 clips × positions, sorted by sign/onset

Usage:
    uv run python src/stage3_analysis/f5384_shape_analysis.py
"""

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

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.dfa_engine import DFAEngine
from ToT_utils import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE      = 5384
NUM_TUBELETS = 8
N_SPATIAL    = 196
HUM_PCT      = 95
REL_THRESH   = 0.25

CFG = {
    "model_flag":       "videomae",
    "layer":            7,
    "device":           "mps" if torch.backends.mps.is_available() else "cpu",
    "classes":          {18: "main", 0: "control"},
    "video_dir":        os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "out_dir":          ROOT / "outputs/analysis/f5384_shape",
    "mass_delta_glob":  "outputs/analysis/**/dfa_mass_delta_vm_c1.parquet",
}


def _resolve_cfg() -> dict:
    sae_path = ROOT / "outputs/sae/sae_layer7_job64.pt"
    dim_mean = ROOT / "outputs/sae/layer7_dim_mean.pt"
    if not sae_path.exists(): raise FileNotFoundError(sae_path)
    if not dim_mean.exists(): raise FileNotFoundError(dim_mean)
    return {"sae_path": str(sae_path), "dim_mean_path": str(dim_mean), "sae_k": 64}


def load_clips(cfg: dict) -> list[tuple[str, int, Path]]:
    matches = list(ROOT.glob(cfg["mass_delta_glob"]))
    assert len(matches) == 1, f"Expected 1 parquet, found: {matches}"
    df = pd.read_parquet(matches[0], columns=["clip_id", "class_id"])
    df = df[df["class_id"].isin(cfg["classes"])]
    video_dir = Path(cfg["video_dir"])
    result = [(str(r.clip_id), int(r.class_id), video_dir / f"{r.clip_id}.webm")
              for r in df.itertuples() if (video_dir / f"{r.clip_id}.webm").exists()]
    log.info(f"  {len(result):,} clips — classes {list(cfg['classes'].keys())}")
    return result


def preprocess(clip_path: Path, processor, num_frames: int, device: str) -> torch.Tensor:
    container = av.open(str(clip_path))
    frames    = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    container.close()
    idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return processor([frames[i] for i in idx], return_tensors="pt")["pixel_values"].to(device)


def tubelet_stats(z: torch.Tensor) -> list[dict]:
    """Per-tubelet stats for feature FEATURE. z: (T, dict_size)."""
    T = z.shape[0]
    assert T == NUM_TUBELETS * N_SPATIAL, (
        f"Token count {T} != {NUM_TUBELETS}×{N_SPATIAL} — verify temporal-major layout"
    )
    feat = z[:, FEATURE].cpu().numpy()                 # (1568,)
    per_tub = feat.reshape(NUM_TUBELETS, N_SPATIAL)   # (8, 196) — temporal-major confirmed
    return [{"position":    t,
             "token_count": int((per_tub[t] > 0).sum()),
             "act_sum":     float(per_tub[t].sum()),
             "act_max":     float(per_tub[t].max())}
            for t in range(NUM_TUBELETS)]


def classify_shape(on_vec: list[bool]) -> tuple[str, int, int, int]:
    """Returns (shape, block_length, onset, offset). onset/offset = -1 if none."""
    positions = [i for i, v in enumerate(on_vec) if v]
    if not positions:
        return "none", 0, -1, -1
    contiguous = (positions[-1] - positions[0] == len(positions) - 1)
    if not contiguous:
        return "other", 0, positions[0], positions[-1]
    onset, offset, blen = positions[0], positions[-1], len(positions)
    if blen == 1:                   return "interior", blen, onset, offset  # explicit singleton override
    if onset == 0 and offset == 7: return "full",     blen, onset, offset
    if onset == 0:                  return "prefix",   blen, onset, offset
    if offset == 7:                 return "suffix",   blen, onset, offset
    return "interior", blen, onset, offset


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def run_stage1(cfg: dict, resolved: dict) -> pd.DataFrame:
    clips  = load_clips(cfg)
    rows   = []
    device = cfg["device"]
    with DFAEngine(cfg["model_flag"], resolved["sae_path"], resolved["dim_mean_path"],
                   layer=cfg["layer"], device=device, sae_k=resolved["sae_k"]) as engine:
        proc = engine._processor
        nf   = engine._num_frames
        for i, (clip_id, class_id, clip_path) in enumerate(clips):
            try:
                pv = preprocess(clip_path, proc, nf, device)
                z  = engine.get_z_pixels(pv)
                for row in tubelet_stats(z):
                    row.update({"clip_id": clip_id, "class_id": class_id})
                    rows.append(row)
            except Exception as exc:
                log.warning(f"SKIP {clip_id}: {exc}")
            if (i + 1) % 20 == 0:
                log.info(f"[{i+1}/{len(clips)}]")
    return pd.DataFrame(rows)[["clip_id","class_id","position","token_count","act_sum","act_max"]]


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def load_dfa_signs(glob: str) -> pd.DataFrame:
    matches = list(ROOT.glob(glob))
    assert len(matches) == 1
    df = pd.read_parquet(matches[0], columns=["clip_id","signed_vec_R"])
    df["dfa_sign"] = df["signed_vec_R"].apply(
        lambda v: float(np.sign(float(np.asarray(v)[FEATURE]))) or 1.0
    )
    return df[["clip_id","dfa_sign"]]


def compute_shape_per_clip(raw: pd.DataFrame, theta: float) -> pd.DataFrame:
    records = []
    for clip_id, grp in raw.groupby("clip_id"):
        grp   = grp.sort_values("position")
        sums  = grp["act_sum"].tolist()
        clip_max = max(sums) if max(sums) > 0 else 1e-10
        abs_on = [s > theta             for s in sums]
        rel_on = [s >= REL_THRESH * clip_max for s in sums]
        shape_a, blen_a, on_a, off_a = classify_shape(abs_on)
        shape_r, blen_r, on_r, off_r = classify_shape(rel_on)
        records.append({
            "clip_id":       clip_id,
            "class_id":      int(grp["class_id"].iloc[0]),
            "shape_abs":     shape_a,
            "shape_rel":     shape_r,
            "block_length":  blen_a,
            "onset":         on_a,
            "offset":        off_a,
            "threshold_agree": shape_a == shape_r,
        })
    return pd.DataFrame(records)


def make_heatmap(raw: pd.DataFrame, shape_df: pd.DataFrame, out_path: Path) -> None:
    df18 = shape_df[(shape_df["class_id"]==18) & (shape_df["shape_abs"]!="none")].copy()
    df18 = df18.sort_values(["dfa_sign","onset"]).reset_index(drop=True)
    mat  = np.zeros((len(df18), NUM_TUBELETS))
    raw18 = raw[raw["class_id"]==18]
    for i, row in df18.iterrows():
        grp = raw18[raw18["clip_id"]==row["clip_id"]].sort_values("position")
        mat[list(df18.index).index(i)] = grp["act_sum"].tolist()

    sign_split = (df18["dfa_sign"] < 0).sum()
    fig, ax = plt.subplots(figsize=(10, max(4, len(df18) * 0.12)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=np.percentile(mat, 99))
    ax.axhline(sign_split - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_xticks(range(8)); ax.set_xticklabels([f"T{i}" for i in range(8)])
    ax.set_ylabel("Clip (neg sign top, pos sign bottom)"); ax.set_xlabel("Tubelet position")
    ax.set_title(f"Feature {FEATURE} — class 18 act_sum per tubelet (none excluded)")
    fig.colorbar(im, ax=ax, label="act_sum")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  heatmap → {out_path}")


def run_stage2(raw: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    # Hum threshold from class-0 rows
    ctrl  = raw[raw["class_id"]==0]["act_sum"]
    theta = float(np.percentile(ctrl, HUM_PCT))
    log.info(f"  θ (class-0 {HUM_PCT}th pct act_sum) = {theta:.6f}")

    shape_df = compute_shape_per_clip(raw, theta)
    signs    = load_dfa_signs(cfg["mass_delta_glob"])
    shape_df = shape_df.merge(signs, on="clip_id", how="left")

    shape_df.to_csv(out_dir / "f5384_shape.csv", index=False)

    # None clips for inspection
    none_clips = shape_df[shape_df["shape_abs"]=="none"]
    none_clips.to_csv(out_dir / "f5384_none_clips.csv", index=False)
    log.info(f"  {len(none_clips)} none clips → f5384_none_clips.csv")

    # Contingency table (class 18 only, abs threshold)
    main = shape_df[(shape_df["class_id"]==18) & (shape_df["shape_abs"]!="none")].copy()
    main["sign_label"] = main["dfa_sign"].apply(lambda s: "+ve" if s > 0 else "-ve")
    ct = pd.crosstab(main["shape_abs"], main["sign_label"], margins=True)
    ct_pct = pd.crosstab(main["shape_abs"], main["sign_label"], normalize="index").round(3) * 100
    ct.to_csv(out_dir / "f5384_contingency.csv")
    log.info(f"\nContingency (class 18, abs threshold):\n{ct}\nRow %:\n{ct_pct}")

    # Onset / offset distributions
    oo = main.groupby("shape_abs")[["onset","offset","block_length"]].describe().round(2)
    oo.to_csv(out_dir / "f5384_onset_offset.csv")
    log.info(f"\nOnset/offset by shape:\n{oo}")

    # Threshold agreement summary
    disagree = shape_df[~shape_df["threshold_agree"]]
    log.info(f"\nThreshold disagreements: {len(disagree)} / {len(shape_df)} clips")

    make_heatmap(raw, shape_df, out_dir / "f5384_heatmap.png")


def main() -> None:
    resolved = _resolve_cfg()
    out_dir: Path = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Stage 1: extraction ===")
    raw = run_stage1(CFG, resolved)
    raw.to_parquet(out_dir / "f5384_raw.parquet", index=False)
    log.info(f"  {len(raw):,} rows → f5384_raw.parquet")

    log.info("=== Stage 2: shape analysis ===")
    run_stage2(raw, CFG, out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
