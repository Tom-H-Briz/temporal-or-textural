"""
Diagnostic: does the perturbation filter (B/C conditions) meaningfully reduce
the set of active SAE features relative to real clips (R)?

Three measurements per class:
  filter_efficiency — fraction of R's active features surviving into B/C
  artefact_check   — fraction of B/C's active features already present in R
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

from sae import BatchTopKSAE
from ToT_utils import SSv2ClipDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ACTIVITY_THRESHOLD_FRAC = 0.01  # feature active if it fires on >1% of tokens
N_TOKENS = 1568
# >15.68 → integer count >= 16
ACTIVE_TOKEN_THRESHOLD = int(N_TOKENS * ACTIVITY_THRESHOLD_FRAC) + 1

NEGATIVE_CONTROL_TEMPLATE = "Pushing [something] so it spins"

CFG = {
    "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
    "sae_checkpoint": str(ROOT / "outputs/sae/sae_layer7_job128.pt"),
    "dim_mean_path": str(ROOT / "outputs/sae/layer7_dim_mean.pt"),
    "perturbation_metadata": str(ROOT / "data/perturbation_metadata.parquet"),
    "validation_path": str(ROOT / "data/ssv2/labels/validation.json"),
    "output_path": str(ROOT / "outputs/analysis/perturbation_filter_diagnostic.parquet"),
    "layer": 7,
    "num_frames": 16,
    "input_dim": 768,
    "nb_concepts": 6144,
    "k": 128,
    "batch_size": 8,
    "num_workers": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def load_clip_metadata(cfg: dict) -> pd.DataFrame:
    """Merge perturbation metadata with validation labels → one row per clip with class_template."""
    meta = pd.read_parquet(cfg["perturbation_metadata"])
    with open(cfg["validation_path"]) as f:
        val = json.load(f)
    val_df = pd.DataFrame(val)[["id", "template"]].rename(
        columns={"id": "clip_id", "template": "class_template"}
    )
    merged = meta.merge(val_df, on="clip_id", how="inner")
    log.info(f"Clip metadata: {len(merged)} clips, {merged['class_template'].nunique()} classes")
    return merged



def _collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def load_model_and_sae(cfg: dict):
    device = cfg["device"]
    log.info("Loading VideoMAE...")
    processor = VideoMAEImageProcessor.from_pretrained(cfg["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(cfg["model_id"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    log.info("Loading SAE checkpoint...")
    dim_mean = torch.load(cfg["dim_mean_path"], weights_only=True).to(device)
    sae = BatchTopKSAE(
        input_shape=cfg["input_dim"],
        nb_concepts=cfg["nb_concepts"],
        top_k=cfg["k"] * N_TOKENS,
        device=device,
    )
    sae.load_state_dict(torch.load(cfg["sae_checkpoint"], weights_only=True, map_location=device))
    return processor, model, sae, dim_mean


def warmup_sae(model, sae, warmup_loader, dim_mean, cfg: dict) -> None:
    """Initialise SAE running_threshold with one batch of clips."""
    device = cfg["device"]
    hook_storage = {}

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_storage["acts"] = hidden.detach()

    handle = model.videomae.encoder.layer[cfg["layer"]].register_forward_hook(_hook)
    sae.train()
    try:
        pixel_values, _ = next(iter(warmup_loader))
        with torch.no_grad():
            model(pixel_values=pixel_values.to(device))
        acts = hook_storage["acts"].detach() - dim_mean
        sae.encode(acts[0].float())
    finally:
        handle.remove()
    sae.eval()
    log.info(f"SAE warmup done. Running threshold: {sae.running_threshold:.5f}")


def compute_mean_activation_vector(activation_matrix: np.ndarray, threshold_frac: float) -> np.ndarray:
    """
    Input: (N_TOKENS, nb_concepts) float array of SAE activations.
    Features firing on fewer than threshold_frac * N_TOKENS tokens are zeroed.
    Returns (nb_concepts,) mean activation across all N_TOKENS tokens.
    """
    n_tokens = activation_matrix.shape[0]
    threshold = int(n_tokens * threshold_frac) + 1
    firing_counts = (activation_matrix > 0).sum(axis=0)   # (nb_concepts,)
    mean_vec = activation_matrix.mean(axis=0)              # (nb_concepts,)
    mean_vec[firing_counts < threshold] = 0.0
    return mean_vec


def compute_magnitude_and_cosine(
    vec_R: np.ndarray, vec_A: np.ndarray, vec_B: np.ndarray, vec_C: np.ndarray
) -> dict:
    """
    Input: four (nb_concepts,) mean activation vectors.
    Returns mean magnitude of non-zero entries and cosine similarities R↔A/B/C.
    """
    def mean_nonzero(v: np.ndarray) -> float:
        nz = v[v > 0]
        return float(nz.mean()) if len(nz) > 0 else np.nan

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan

    return {
        "mean_magnitude_R": mean_nonzero(vec_R),
        "mean_magnitude_A": mean_nonzero(vec_A),
        "mean_magnitude_B": mean_nonzero(vec_B),
        "mean_magnitude_C": mean_nonzero(vec_C),
        "cosine_sim_RA": cosine_sim(vec_R, vec_A),
        "cosine_sim_RB": cosine_sim(vec_R, vec_B),
        "cosine_sim_RC": cosine_sim(vec_R, vec_C),
    }


def compute_active_features(
    model, sae, clip_paths: list[Path], clip_ids: list[str],
    processor, dim_mean, cfg: dict, desc: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Returns:
      active_masks — {clip_id: bool array [nb_concepts]} active on >= ACTIVE_TOKEN_THRESHOLD tokens
      mean_vecs    — {clip_id: float array [nb_concepts]} mean activation vector per clip
    """
    device = cfg["device"]
    hook_storage = {}

    def _hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_storage["acts"] = hidden.detach()

    dataset = SSv2ClipDataset(clip_paths, processor, cfg["num_frames"], labels=clip_ids)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=_collate,
        pin_memory=(device == "cuda"),
    )

    handle = model.videomae.encoder.layer[cfg["layer"]].register_forward_hook(_hook)
    active_masks: dict[str, np.ndarray] = {}
    mean_vecs: dict[str, np.ndarray] = {}
    try:
        with torch.no_grad():
            for pixel_values, ids in tqdm(loader, desc=f"  {desc}"):
                pixel_values = pixel_values.to(device)
                with torch.autocast(device_type=device, enabled=(device == "cuda")):
                    model(pixel_values=pixel_values)
                acts = hook_storage["acts"]           # [B, N_TOKENS, input_dim]
                B = acts.shape[0]
                tokens = (acts.reshape(B * N_TOKENS, cfg["input_dim"]) - dim_mean).float()
                _, z = sae.encode(tokens)             # [B*N_TOKENS, nb_concepts]
                z = z.reshape(B, N_TOKENS, cfg["nb_concepts"])
                counts = (z > 0).sum(dim=1)           # [B, nb_concepts]
                active = (counts >= ACTIVE_TOKEN_THRESHOLD).cpu().numpy()
                z_np = z.cpu().numpy()                # [B, N_TOKENS, nb_concepts]
                for j, cid in enumerate(ids):
                    active_masks[cid] = active[j]
                    mean_vecs[cid] = compute_mean_activation_vector(z_np[j], ACTIVITY_THRESHOLD_FRAC)
    finally:
        handle.remove()
    return active_masks, mean_vecs


def compute_clip_fractions(
    clip_ids: list[str],
    r_feats: dict[str, np.ndarray],
    a_feats: dict[str, np.ndarray],
    b_feats: dict[str, np.ndarray],
    c_feats: dict[str, np.ndarray],
) -> list[dict]:
    records = []
    for cid in clip_ids:
        R = r_feats[cid]
        A = a_feats[cid]
        B = b_feats[cid]
        C = c_feats[cid]
        r_n = int(R.sum())
        a_n = int(A.sum())
        b_n = int(B.sum())
        c_n = int(C.sum())

        if r_n == 0:
            log.warning(f"Clip {cid}: |R_active|=0 — excluding from efficiency/artefact metrics")
        if a_n == 0:
            log.warning(f"Clip {cid}: |A_active|=0 — excluding from artefact_check_A")
        if b_n == 0:
            log.warning(f"Clip {cid}: |B_active|=0 — excluding from artefact_check_B")
        if c_n == 0:
            log.warning(f"Clip {cid}: |C_active|=0 — excluding from artefact_check_C")

        ra = int((R & A).sum())
        rb = int((R & B).sum())
        rc = int((R & C).sum())

        records.append({
            "clip_id": cid,
            "r_active_count": r_n,
            "a_active_count": a_n,
            "b_active_count": b_n,
            "c_active_count": c_n,
            "filter_efficiency_A": ra / r_n if r_n > 0 else np.nan,
            "filter_efficiency_B": rb / r_n if r_n > 0 else np.nan,
            "filter_efficiency_C": rc / r_n if r_n > 0 else np.nan,
            "artefact_check_A": ra / a_n if a_n > 0 else np.nan,
            "artefact_check_B": rb / b_n if b_n > 0 else np.nan,
            "artefact_check_C": rc / c_n if c_n > 0 else np.nan,
        })
    return records


ALL_METRICS = [
    "filter_efficiency_A", "filter_efficiency_B", "filter_efficiency_C",
    "artefact_check_A", "artefact_check_B", "artefact_check_C",
    "r_active_count", "a_active_count", "b_active_count", "c_active_count",
    "mean_magnitude_R", "mean_magnitude_A", "mean_magnitude_B", "mean_magnitude_C",
    "cosine_sim_RA", "cosine_sim_RB", "cosine_sim_RC",
]


def build_class_summary(clip_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-class summary: one row per (class_label, metric)."""
    records = []
    for class_label, group in clip_df.groupby("class_label"):
        for m in ALL_METRICS:
            col = group[m].dropna()
            records.append({
                "class_label": class_label,
                "metric": m,
                "mean": col.mean() if len(col) else np.nan,
                "median": col.median() if len(col) else np.nan,
                "iqr": col.quantile(0.75) - col.quantile(0.25) if len(col) else np.nan,
                "n_valid": len(col),
            })
    return pd.DataFrame(records)


def print_summary(clip_df: pd.DataFrame) -> None:
    def iqr(s: pd.Series) -> float:
        return s.quantile(0.75) - s.quantile(0.25)

    print(f"\n{'='*68}")
    print("PERTURBATION FILTER DIAGNOSTIC")
    print(f"Activity threshold: >{N_TOKENS * ACTIVITY_THRESHOLD_FRAC:.2f} tokens (≥{ACTIVE_TOKEN_THRESHOLD})")
    print(f"{'='*68}")

    for label, group in clip_df.groupby("class_label"):
        display = str(label) + "  [negative control]" if label == NEGATIVE_CONTROL_TEMPLATE else str(label)
        print(f"\nClass: {display}  (n={len(group)})")
        print(f"{'-'*68}")
        print(f"{'Metric':<24} {'Mean':>8} {'Median':>8} {'IQR':>8}  {'n_valid':>7}")
        print(f"{'-'*68}")
        for m in ALL_METRICS:
            col = group[m].dropna()
            if col.empty:
                print(f"{m:<24} {'N/A':>8} {'N/A':>8} {'N/A':>8}  {'0':>7}")
                continue
            print(f"{m:<24} {col.mean():>8.3f} {col.median():>8.3f} {iqr(col):>8.3f}  {len(col):>7}")


def main() -> None:
    cfg = CFG
    log.info(f"Device: {cfg['device']}")
    log.info(f"Active token threshold: >= {ACTIVE_TOKEN_THRESHOLD} ({ACTIVITY_THRESHOLD_FRAC*100}% of {N_TOKENS})")

    clip_meta = load_clip_metadata(cfg)
    if "path_A" not in clip_meta.columns:
        raise RuntimeError("path_A column missing — run perturbationA.py first")

    clip_ids = clip_meta["clip_id"].tolist()
    r_paths = [Path(p) for p in clip_meta["source_path"]]
    a_paths = [Path(p) for p in clip_meta["path_A"]]
    b_paths = [Path(p) for p in clip_meta["path_B"]]
    c_paths = [Path(p) for p in clip_meta["path_C"]]

    processor, model, sae, dim_mean = load_model_and_sae(cfg)

    log.info("Warming up SAE running threshold...")
    warmup_ds = SSv2ClipDataset(r_paths[: cfg["batch_size"]], processor, cfg["num_frames"], labels=clip_ids[: cfg["batch_size"]])
    warmup_loader = DataLoader(warmup_ds, batch_size=cfg["batch_size"], collate_fn=_collate)
    warmup_sae(model, sae, warmup_loader, dim_mean, cfg)

    template_map = dict(zip(clip_meta["clip_id"], clip_meta["class_template"]))

    log.info("Processing R (real clips)...")
    r_feats, r_vecs = compute_active_features(model, sae, r_paths, clip_ids, processor, dim_mean, cfg, "R")

    log.info("Processing A (still/midpoint)...")
    a_feats, a_vecs = compute_active_features(model, sae, a_paths, clip_ids, processor, dim_mean, cfg, "A")

    log.info("Processing B (first/last)...")
    b_feats, b_vecs = compute_active_features(model, sae, b_paths, clip_ids, processor, dim_mean, cfg, "B")

    log.info("Processing C (shuffle)...")
    c_feats, c_vecs = compute_active_features(model, sae, c_paths, clip_ids, processor, dim_mean, cfg, "C")

    log.info("Computing per-clip overlap fractions...")
    records = compute_clip_fractions(clip_ids, r_feats, a_feats, b_feats, c_feats)
    results_df = pd.DataFrame(records)
    results_df["class_template"] = results_df["clip_id"].map(template_map)

    log.info("Computing magnitude and cosine diagnostics...")
    mc_records = []
    for cid in clip_ids:
        mc = compute_magnitude_and_cosine(r_vecs[cid], a_vecs[cid], b_vecs[cid], c_vecs[cid])
        mc["clip_id"] = cid
        mc["class_label"] = template_map[cid]
        mc_records.append(mc)
    mc_df = pd.DataFrame(mc_records)

    clip_df = (
        results_df.rename(columns={"class_template": "class_label"})
        .merge(mc_df.drop(columns=["class_label"]), on="clip_id")
    )

    out_path = Path(cfg["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clip_path = out_path.parent / "mag_cos_diag_clip.csv"
    clip_df.to_csv(clip_path, index=False)
    log.info(f"Saved → {clip_path}")

    class_df = build_class_summary(clip_df)
    class_path = out_path.parent / "mag_cos_diag_class.csv"
    class_df.to_csv(class_path, index=False)
    log.info(f"Saved → {class_path}")

    print_summary(clip_df)


if __name__ == "__main__":
    main()
