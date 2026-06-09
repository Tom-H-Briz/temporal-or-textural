"""
Measures VideoMAE per-class accuracy on the 5 target classes with a trained SAE
spliced into layer 7. Compares multiple SAE configs against the perturb_accuracy baseline.

Output: outputs/spliced_accuracy/
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from sae import BatchTopKSAE
from class_selection import CFG, compute_accuracy_df, save_charts
from ToT_utils import (
    _strip_brackets, load_metadata, make_sae_splice_hook,
    run_inference, SSv2ClipDataset,
)

# SAEs to compare — add or remove entries here to extend the sweep
SAE_CONFIGS = [
    {"label": "128_16x",  "k": 128},
]

LAYER = 7
OUTPUT_DIR = ROOT / "outputs" / "spliced_accuracy"
BASELINE_DIR = ROOT / "outputs" / "stage1_perturb_accuracy"
SAE_DIR = ROOT / "outputs" / "sae"
DIM_MEAN_PATH = SAE_DIR / "layer7_dim_mean.pt"

def load_sae(cfg: dict, dim_mean: torch.Tensor, device: str) -> BatchTopKSAE:
    """Load a checkpoint, warm up running_threshold, return in eval mode."""
    checkpoint = SAE_DIR / f"sae_layer7_job{cfg['label']}.pt"
    _ckpt = torch.load(checkpoint, weights_only=True, map_location=device)
    if isinstance(_ckpt, dict) and "sae_state_dict" in _ckpt:
        _ckpt = _ckpt["sae_state_dict"]
    nb_concepts = _ckpt["dictionary._weights"].shape[0]
    sae = BatchTopKSAE(
        input_shape=768,
        nb_concepts=nb_concepts,
        top_k=cfg["k"] * 1568,
        device=device,
    )
    sae.load_state_dict(_ckpt)

    # running_threshold is not persisted — initialise with one dummy forward pass
    sae.train()
    dummy = torch.zeros(1568, 768, device=device)
    with torch.no_grad():
        sae.encode((dummy - dim_mean).float())
    sae.eval()
    return sae


def build_loader(paths: list[Path], labels: list[int], processor) -> DataLoader:
    dataset = SSv2ClipDataset(paths, processor, CFG["num_frames"], labels=labels)
    return DataLoader(
        dataset, batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"], pin_memory=False,
    )


TARGET_CLASSES = {
    "Pushing something from right to left",
    "Pushing something from left to right",
    "Tearing something just a little bit",
    "Tearing something into two pieces",
    "Pushing something so it spins",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = CFG["device"]

    dim_mean = torch.load(DIM_MEAN_PATH, weights_only=True).to(device)

    print("Loading metadata...")
    label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])
    clip_label: dict[str, int] = {
        str(c["id"]): label_map[_strip_brackets(c["template"])]
        for c in clips
        if _strip_brackets(c["template"]) in TARGET_CLASSES
    }
    print(f"  {len(clip_label):,} clips in target classes")

    meta = pd.read_parquet(ROOT / "data" / "perturbation_metadata.parquet")
    meta["clip_id"] = meta["clip_id"].astype(str)
    meta = meta[meta["clip_id"].isin(clip_label)]
    rows = list(meta.itertuples())

    video_dir = Path(CFG["video_dir"])
    paths_A = [video_dir / f"{str(r.clip_id)}.webm" for r in rows]
    paths_B = [Path(str(r.path_B)) for r in rows]
    paths_C = [Path(str(r.path_C)) for r in rows]
    labels  = [clip_label[str(r.clip_id)] for r in rows]

    print(f"Loading model: {CFG['model_id']}")
    processor = VideoMAEImageProcessor.from_pretrained(CFG["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(CFG["model_id"]).to(device)
    splice_layer = model.videomae.encoder.layer[LAYER]

    all_results: list[dict] = []

    for sae_cfg in SAE_CONFIGS:
        label = sae_cfg["label"]
        print(f"\n--- SAE k={label} ---")
        sae = load_sae(sae_cfg, dim_mean, device)
        hook_handle = splice_layer.register_forward_hook(
            make_sae_splice_hook(sae, dim_mean)
        )

        for tag, paths in [("A", paths_A), ("B", paths_B), ("C", paths_C)]:
            tag_name = {"A": "original", "B": "first/last", "C": "shuffle"}[tag]
            print(f"  {tag} ({tag_name})")
            loader = build_loader(paths, labels, processor)
            preds, true_labels = run_inference(model, loader, device)

            overall = sum(p == l for p, l in zip(preds, true_labels)) / len(preds)

            df = compute_accuracy_df(preds, true_labels, id2template)
            df = df[df["total"] > 0].reset_index(drop=True)
            print(f"    Clip-weighted: {overall:.4f}  |  Mean per-class: {df['accuracy'].mean():.4f}")

            csv_path = OUTPUT_DIR / f"per_class_accuracy_sae{label}_{tag}.csv"
            df.to_csv(csv_path, index=False)

            charts_dir = OUTPUT_DIR / f"sae{label}_{tag}"
            charts_dir.mkdir(exist_ok=True)
            save_charts(df, charts_dir)

            for _, row in df.iterrows():
                all_results.append({
                    "sae": label, "condition": tag,
                    "class_id": row["class_id"], "template": row["template"],
                    "accuracy": row["accuracy"], "total": row["total"],
                })

        hook_handle.remove()

    # Wide comparison table: one row per class, columns per (sae, condition)
    comp = pd.DataFrame(all_results)
    comp = comp.pivot_table(
        index=["class_id", "template"], columns=["sae", "condition"],
        values="accuracy",
    )
    comp.columns = [f"sae{s}_{c}" for s, c in comp.columns]
    comp = comp.reset_index().sort_values("class_id")
    comp.to_csv(OUTPUT_DIR / "comparison_16x.csv", index=False)
    print(f"\nComparison saved → {OUTPUT_DIR / 'comparison_sae_ep7.csv'}")


if __name__ == "__main__":
    main()
