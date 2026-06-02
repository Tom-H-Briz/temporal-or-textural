"""
Runs top-1 inference on the SSv2 validation set with VideoMAE and computes per-class accuracy.
Saves a CSV of all 174 classes and top-10/bottom-10 bar charts to outputs/stage1_class_selection/.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import VideoMAEImageProcessor

from ToT_utils import MODEL_ID, NUM_CLASSES, NUM_FRAMES, _strip_brackets, load_metadata, run_inference

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = {
    "model_id": MODEL_ID,
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "validation_path": ROOT / "data/ssv2/labels/validation.json",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "output_dir": ROOT / "outputs/stage1_class_selection",
    "batch_size": 4,
    "num_workers": 0,
    "num_frames": NUM_FRAMES,
    "device": (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ),
}


def build_dataloader(
    clips: list[dict], label_map: dict[str, int], processor: VideoMAEImageProcessor
) -> DataLoader:
    dataset = SSv2Dataset(
        clips=clips,
        video_dir=Path(CFG["video_dir"]),
        label_map=label_map,
        processor=processor,
        num_frames=CFG["num_frames"],
    )
    return DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        pin_memory=False,
    )


def compute_accuracy_df(
    preds: list[int], labels: list[int], id2template: dict[int, str]
) -> pd.DataFrame:
    correct: dict[int, int] = {i: 0 for i in range(174)}
    total: dict[int, int] = {i: 0 for i in range(174)}

    for pred, label in zip(preds, labels):
        total[label] += 1
        if pred == label:
            correct[label] += 1

    rows = [
        {
            "class_id": cid,
            "template": id2template.get(cid, ""),
            "correct": correct[cid],
            "total": total[cid],
            "accuracy": correct[cid] / total[cid] if total[cid] > 0 else float("nan"),
        }
        for cid in range(174)
    ]

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def save_charts(df: pd.DataFrame, output_dir: Path) -> None:
    for filename, title, subset in [
        ("top10_accuracy.png", "Top-10 Classes by Accuracy", df.head(10)),
        ("bottom10_accuracy.png", "Bottom-10 Classes by Accuracy", df.tail(10).iloc[::-1]),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(subset["template"], subset["accuracy"])
        ax.set_xlabel("Top-1 Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        plt.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)


def main() -> None:
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])
    print(f"  {len(clips):,} validation clips, {len(label_map)} classes")

    print(f"Loading model: {CFG['model_id']}")
    processor = VideoMAEImageProcessor.from_pretrained(CFG["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(CFG["model_id"]).to(CFG["device"])

    dataloader = build_dataloader(clips, label_map, processor)
    preds, labels = run_inference(model, dataloader, CFG["device"])

    overall = sum(p == l for p, l in zip(preds, labels)) / len(preds)
    print(f"\nOverall top-1 accuracy: {overall:.4f}")

    df = compute_accuracy_df(preds, labels, id2template)

    csv_path = output_dir / "per_class_accuracy.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    save_charts(df, output_dir)
    print(f"Saved charts to {output_dir}/")


if __name__ == "__main__":
    main()
