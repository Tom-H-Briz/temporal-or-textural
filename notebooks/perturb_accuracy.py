"""
Evaluates VideoMAE per-class accuracy on perturbed clips (B = first/last, C = shuffle).

Reuses: CFG, _strip_brackets, load_metadata, run_inference,
        compute_accuracy_df, save_charts  from class_selection.py.

Output:
  outputs/stage1_perturb_accuracy/per_class_accuracy_B.csv
  outputs/stage1_perturb_accuracy/per_class_accuracy_C.csv
  outputs/stage1_perturb_accuracy/comparison.csv   (B vs C side-by-side)
  outputs/stage1_perturb_accuracy/{B,C}/top10_accuracy.png
  outputs/stage1_perturb_accuracy/{B,C}/bottom10_accuracy.png
"""

import sys
from pathlib import Path

import av
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from class_selection import (
    CFG,
    _strip_brackets,
    compute_accuracy_df,
    load_metadata,
    run_inference,
    save_charts,
)

TARGET_CLASSES = {
    "Pushing something from right to left",
    "Pushing something from left to right",
    "Tearing something just a little bit",
    "Tearing something into two pieces",
    "Pushing something so it spins",
}

PERTURB_META = ROOT / "data" / "perturbation_metadata.parquet"
OUTPUT_DIR = ROOT / "outputs" / "stage1_perturb_accuracy"


class PerturbedDataset(Dataset):
    """Loads pre-built perturbed clips from explicit paths."""

    def __init__(self, items: list[tuple[Path, int]], processor, num_frames: int) -> None:
        self.items = items
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        container = av.open(str(path))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()

        n = len(frames)
        indices = torch.linspace(0, n - 1, self.num_frames).long().tolist()
        sampled = [frames[i] for i in indices]

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label


def build_loader(items: list[tuple[Path, int]], processor) -> DataLoader:
    dataset = PerturbedDataset(items, processor, CFG["num_frames"])
    return DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])

    # clip_id (str) → label_id for our 5 target classes only
    clip_label: dict[str, int] = {
        str(c["id"]): label_map[_strip_brackets(c["template"])]
        for c in clips
        if _strip_brackets(c["template"]) in TARGET_CLASSES
    }
    print(f"  {len(clip_label):,} clips in target classes")

    meta = pd.read_parquet(PERTURB_META)
    meta["clip_id"] = meta["clip_id"].astype(str)
    meta = meta[meta["clip_id"].isin(clip_label)]
    print(f"  {len(meta):,} clips in perturbation metadata")

    video_dir = Path(CFG["video_dir"])
    items_A = [(video_dir / f"{r.clip_id}.webm",  clip_label[r.clip_id]) for r in meta.itertuples()]
    items_B = [(Path(r.path_B), clip_label[r.clip_id]) for r in meta.itertuples()]
    items_C = [(Path(r.path_C), clip_label[r.clip_id]) for r in meta.itertuples()]

    print(f"Loading model: {CFG['model_id']}")
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(CFG["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(CFG["model_id"]).to(CFG["device"])

    acc_dfs: dict[str, pd.DataFrame] = {}

    TAGS = [("A", "original"), ("B", "first/last"), ("C", "shuffle")]
    for tag, items in [("A", items_A), ("B", items_B), ("C", items_C)]:
        label = dict(TAGS)[tag]
        print(f"\n{tag} ({label}) — {len(items):,} clips")

        preds, labels = run_inference(model, build_loader(items, processor))

        overall = sum(p == l for p, l in zip(preds, labels)) / len(preds)
        print(f"  Overall accuracy: {overall:.4f}")

        df = compute_accuracy_df(preds, labels, id2template)
        df = df[df["total"] > 0].reset_index(drop=True)

        csv_path = OUTPUT_DIR / f"per_class_accuracy_{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        charts_dir = OUTPUT_DIR / tag
        charts_dir.mkdir(exist_ok=True)
        save_charts(df, charts_dir)

        acc_dfs[tag] = df

    # side-by-side comparison: A (original) vs B vs C
    merged = (
        acc_dfs["A"][["class_id", "template", "total", "accuracy"]].rename(columns={"accuracy": "accuracy_A"})
        .merge(acc_dfs["B"][["class_id", "accuracy"]].rename(columns={"accuracy": "accuracy_B"}), on="class_id")
        .merge(acc_dfs["C"][["class_id", "accuracy"]].rename(columns={"accuracy": "accuracy_C"}), on="class_id")
    )
    merged["delta_B_minus_A"] = merged["accuracy_B"] - merged["accuracy_A"]
    merged["delta_C_minus_A"] = merged["accuracy_C"] - merged["accuracy_A"]
    merged = merged.sort_values("accuracy_A", ascending=False).reset_index(drop=True)

    comp_path = OUTPUT_DIR / "comparison.csv"
    merged.to_csv(comp_path, index=False)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
