"""
Runs top-1 inference on the SSv2 validation set with VideoMAE and computes per-class accuracy.
Saves a CSV of all 174 classes and top-10/bottom-10 bar charts to outputs/stage1_class_selection/.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import av
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CFG = {
    "model_id": "MCG-NJU/videomae-base-finetuned-ssv2",
    "labels_path": ROOT / "data/ssv2/labels/labels.json",
    "validation_path": ROOT / "data/ssv2/labels/validation.json",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "output_dir": ROOT / "outputs/stage1_class_selection",
    "batch_size": 4,
    "num_workers": 0,
    "num_frames": 16,
    "device": (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ),
}


def _strip_brackets(template: str) -> str:
    return template.replace("[", "").replace("]", "")


class SSv2Dataset(Dataset):
    def __init__(
        self,
        clips: list[dict],
        video_dir: Path,
        label_map: dict[str, int],
        processor: VideoMAEImageProcessor,
        num_frames: int,
    ) -> None:
        self.clips = clips
        self.video_dir = video_dir
        self.label_map = label_map
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clip = self.clips[idx]
        path = self.video_dir / f"{clip['id']}.webm"
        label = self.label_map[_strip_brackets(clip["template"])]

        container = av.open(str(path))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()

        n = len(frames)
        indices = torch.linspace(0, n - 1, self.num_frames).long().tolist()
        sampled = [frames[i] for i in indices]

        pixel_values = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label


def load_metadata(
    labels_path: str, validation_path: str
) -> tuple[dict[str, int], list[dict], dict[int, str]]:
    with open(labels_path) as f:
        raw: dict[str, str] = json.load(f)

    # labels.json comes in two formats depending on dataset release:
    #   {"template name": "0", ...}  →  name-keyed
    #   {"0": "template name", ...}  →  index-keyed
    first_key = next(iter(raw))
    if first_key.isdigit():
        label_map = {name: int(idx) for idx, name in raw.items()}
        id2template = {int(idx): name for idx, name in raw.items()}
    else:
        label_map = {template: int(idx) for template, idx in raw.items()}
        id2template = {int(idx): template for template, idx in raw.items()}

    with open(validation_path) as f:
        all_clips: list[dict] = json.load(f)

    clips = [c for c in all_clips if _strip_brackets(c["template"]) in label_map]
    n_dropped = len(all_clips) - len(clips)
    if n_dropped:
        print(f"  Warning: dropped {n_dropped} clips with templates not in labels.json")

    return label_map, clips, id2template


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


def run_inference(
    model: VideoMAEForVideoClassification, dataloader: DataLoader
) -> tuple[list[int], list[int]]:
    all_preds: list[int] = []
    all_labels: list[int] = []
    model.eval()

    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc="Inference"):
            pixel_values = pixel_values.to(CFG["device"])
            preds = model(pixel_values=pixel_values).logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


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
    preds, labels = run_inference(model, dataloader)

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
