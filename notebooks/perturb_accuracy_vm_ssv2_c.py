"""
VM per-class accuracy under the ORIGINAL C shuffle — post-fix rerun, SSv2 val set.

C shuffles ALL native frames (apply_shuffle, full permutation) BEFORE sampling,
then uniformly samples 16 frames — the exact mechanism perturb_accuracy_vm.py
used (per_class_accuracy_VM_C.csv, dated 26/06, pre-30/07 backbone fix).
Distinct from C1 (perturb_accuracy_vm_ssv2.py): C1 samples the 16 frames FIRST,
then shuffles them as consecutive PAIRS — introduced because full-permutation-
before-sampling can put two temporally-unrelated frames in the same tubelet
token, tearing it. No post-fix run of the original C existed until now.

Output:
    outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_C.csv

Usage:
    uv run python notebooks/perturb_accuracy_vm_ssv2_c.py
"""

import os
import sys
from pathlib import Path

import av
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(Path(__file__).parent))

from perturbation import apply_shuffle
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, _strip_brackets, load_metadata

CFG = {
    "model_name":      "videomae",
    "labels_path":     os.environ.get("LABELS_PATH",     str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR",        str(ROOT / "data/ssv2/20bn-something-something-v2")),
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "output_dir":      str(ROOT / "outputs/stage1_class_selection_VM_ssv2"),
}
_model_cfg        = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"] = _model_cfg["num_frames"]  # 16


class CDataset(Dataset):
    """Shuffle-then-sample, exactly as perturb_accuracy_vm.py's condition C:
    apply_shuffle on ALL native frames, THEN torch.linspace-sample 16 — NOT
    C1's sample-then-shuffle-pairs (perturb_accuracy_vm_ssv2.py)."""

    def __init__(self, clip_paths, clip_ids, labels, processor, num_frames):
        self.clip_paths, self.clip_ids, self.labels = clip_paths, clip_ids, labels
        self.processor, self.num_frames = processor, num_frames

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        container = av.open(str(self.clip_paths[idx]))
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        container.close()
        seed = int(self.clip_ids[idx]) % 2**32
        frames = apply_shuffle(frames, seed)
        indices = torch.linspace(0, len(frames) - 1, self.num_frames).long().tolist()
        sampled = [frames[i] for i in indices]
        pv = self.processor(sampled, return_tensors="pt")["pixel_values"].squeeze(0)
        return pv, self.labels[idx]


def load_clips(cfg: dict) -> tuple[list[Path], list[str], list[int], dict[int, str]]:
    label_map, clips, id2template = load_metadata(cfg["labels_path"], cfg["validation_path"])
    video_dir = Path(cfg["video_dir"])
    paths, clip_ids, labels = [], [], []
    for c in clips:
        template = _strip_brackets(c["template"])
        if template not in label_map:
            continue
        path = video_dir / f"{c['id']}.webm"
        if not path.exists():
            continue
        paths.append(path)
        clip_ids.append(c["id"])
        labels.append(label_map[template])
    print(f"  {len(paths):,} clips")
    return paths, clip_ids, labels, id2template


def save_csv(preds: list[int], labels: list[int], id2template: dict[int, str], out_path: Path) -> None:
    correct = {i: 0 for i in range(174)}
    total = {i: 0 for i in range(174)}
    for pred, label in zip(preds, labels):
        total[label] += 1
        correct[label] += int(pred == label)
    rows = [{"class_id": cid, "template": id2template[cid],
             "correct": correct[cid], "total": total[cid],
             "accuracy": correct[cid] / total[cid] if total[cid] else float("nan")}
            for cid in range(174) if total[cid] > 0]
    pd.DataFrame(rows).sort_values("accuracy", ascending=False).to_csv(out_path, index=False)
    overall = sum(correct.values()) / sum(total.values())
    print(f"  Overall top-1: {overall:.4f}  -> {out_path.name}")


def main() -> None:
    device = CFG["device"]
    print(f"Device: {device}  Model: {CFG['model_name']}  Condition: C (post-fix rerun)")

    clip_paths, clip_ids, labels, id2template = load_clips(CFG)

    model_cfg = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], "ssv2")]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(device).eval()

    dataset = CDataset(clip_paths, clip_ids, labels, processor, CFG["num_frames"])
    loader = DataLoader(dataset, batch_size=CFG["batch_size"],
                        num_workers=CFG["num_workers"], pin_memory=True)
    preds = []
    with torch.no_grad():
        for pixel_values, _ in tqdm(loader, desc="Condition C"):
            preds.extend(model(pixel_values=pixel_values.to(device)).logits.argmax(dim=-1).cpu().tolist())

    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(preds, labels, id2template, out_dir / "per_class_accuracy_VM_ssv2_C.csv")


if __name__ == "__main__":
    main()
