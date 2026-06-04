"""
Evaluates VideoMAE per-class accuracy across four conditions:
  R — real clips
  A — single midpoint frame (still)
  B — first/last frame freeze
  C — frame shuffle

Output:
  outputs/stage1_perturb_accuracy/per_class_accuracy_{R,A,B,C}.csv
  outputs/stage1_perturb_accuracy/comparison.csv   (all four, R as baseline)
  outputs/stage1_perturb_accuracy/{R,A,B,C}/top10_accuracy.png
  outputs/stage1_perturb_accuracy/{R,A,B,C}/bottom10_accuracy.png
"""

import sys
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from class_selection import CFG, compute_accuracy_df, save_charts
from ToT_utils import SSv2ClipDataset, _strip_brackets, load_metadata, run_inference

TARGET_CLASSES = {
    "Pushing something from right to left",
    "Pushing something from left to right",
    "Tearing something just a little bit",
    "Tearing something into two pieces",
    "Pushing something so it spins",
}

PERTURB_META = ROOT / "data" / "perturbation_metadata.parquet"
OUTPUT_DIR = ROOT / "outputs" / "stage1_perturb_accuracy"

TAG_LABEL = {"R": "real", "A": "still", "B": "first/last", "C": "shuffle"}


def build_loader(paths: list[Path], labels: list[int], processor) -> DataLoader:
    dataset = SSv2ClipDataset(paths, processor, CFG["num_frames"], labels=labels)
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

    if "path_A" not in meta.columns:
        raise RuntimeError("path_A column missing — run perturbationA.py first")

    video_dir = Path(CFG["video_dir"])
    rows = list(meta.itertuples())

    paths_R = [video_dir / f"{r.clip_id}.webm" for r in rows]
    paths_A = [Path(r.path_A) for r in rows]
    paths_B = [Path(r.path_B) for r in rows]
    paths_C = [Path(r.path_C) for r in rows]
    labels  = [clip_label[r.clip_id] for r in rows]

    print(f"Loading model: {CFG['model_id']}")
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(CFG["model_id"])
    model = VideoMAEForVideoClassification.from_pretrained(CFG["model_id"]).to(CFG["device"])

    acc_dfs: dict[str, pd.DataFrame] = {}

    for tag, paths in [("R", paths_R), ("A", paths_A), ("B", paths_B), ("C", paths_C)]:
        print(f"\n{tag} ({TAG_LABEL[tag]}) — {len(paths):,} clips")

        preds, true_labels = run_inference(model, build_loader(paths, labels, processor), CFG["device"])

        overall = sum(p == l for p, l in zip(preds, true_labels)) / len(preds)
        print(f"  Overall accuracy: {overall:.4f}")

        df = compute_accuracy_df(preds, true_labels, id2template)
        df = df[df["total"] > 0].reset_index(drop=True)

        csv_path = OUTPUT_DIR / f"per_class_accuracy_{tag}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        charts_dir = OUTPUT_DIR / tag
        charts_dir.mkdir(exist_ok=True)
        save_charts(df, charts_dir)

        acc_dfs[tag] = df

    merged = (
        acc_dfs["R"][["class_id", "template", "total", "accuracy"]].rename(columns={"accuracy": "accuracy_R"})
        .merge(acc_dfs["A"][["class_id", "accuracy"]].rename(columns={"accuracy": "accuracy_A"}), on="class_id")
        .merge(acc_dfs["B"][["class_id", "accuracy"]].rename(columns={"accuracy": "accuracy_B"}), on="class_id")
        .merge(acc_dfs["C"][["class_id", "accuracy"]].rename(columns={"accuracy": "accuracy_C"}), on="class_id")
    )
    merged["delta_A_minus_R"] = merged["accuracy_A"] - merged["accuracy_R"]
    merged["delta_B_minus_R"] = merged["accuracy_B"] - merged["accuracy_R"]
    merged["delta_C_minus_R"] = merged["accuracy_C"] - merged["accuracy_R"]
    merged = merged.sort_values("accuracy_R", ascending=False).reset_index(drop=True)

    comp_path = OUTPUT_DIR / "comparison.csv"
    merged.to_csv(comp_path, index=False)
    print(f"\nComparison saved: {comp_path}")


if __name__ == "__main__":
    main()
