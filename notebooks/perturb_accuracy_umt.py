"""
UMT per-class accuracy under conditions R/C/A — SSv2 val set (full).

Same conditions and code as perturb_accuracy_tf_kinetics.py (TF convention — UMT's
tokens are single frames, tubelet_size=1): R real; C all native frames permuted
pre-sampling (seed _deterministic_seed(clip_id)); A midpoint frame repeated
pre-sampling. Sampler is UMT's own 'middle' protocol via get_frame_sampler.

Outputs (outputs/stage1_class_selection_UMT_ssv2/):
  per_class_accuracy_UMT_ssv2_R.csv / _C.csv / _A.csv
  comparison.csv   (all three, wide format, R as baseline)

Usage: uv run python notebooks/perturb_accuracy_umt.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(Path(__file__).parent))

from ToT_utils import CHECKPOINT_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY, get_frame_sampler
from perturb_accuracy_tf_kinetics import merge_conditions, run_condition
from perturb_accuracy_vm_kinetics import save_csv
from perturb_accuracy_vm_ssv2 import load_ssv2_clips

_ssv2 = DATASET_REGISTRY["ssv2"]
CFG = {
    "model_name":      "umt",
    "dataset_name":    "ssv2",
    "labels_path":     os.environ.get("LABELS_PATH",     str(_ssv2["labels_path"])),
    "validation_path": os.environ.get("VALIDATION_PATH", str(_ssv2["validation_path"])),
    "video_dir":       os.environ.get("VIDEO_DIR",       str(_ssv2["video_dir"])),
    "max_clips":       int(os.environ.get("MAX_CLIPS", 0)),   # 0 = full val set; >0 local smoke test only
    "batch_size":      8,
    "num_workers":     4,
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir":      str(ROOT / "outputs" / "stage1_class_selection_UMT_ssv2"),
}
_model_cfg           = MODEL_REGISTRY[CFG["model_name"]]
CFG["num_frames"]    = _model_cfg["num_frames"]                               # 12
CFG["frame_sampler"] = get_frame_sampler(CFG["dataset_name"], _model_cfg)    # UMT 'middle'


def main() -> None:
    device = CFG["device"]
    print(f"Device: {device}  Model: {CFG['model_name']}  Dataset: {CFG['dataset_name']}")
    model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
    checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], CFG["dataset_name"])]
    processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
    model      = model_cfg["model_class"].from_pretrained(checkpoint)
    model.to(device).eval()

    clip_paths, clip_ids, labels, id2label = load_ssv2_clips(CFG)
    out_dir = Path(CFG["output_dir"])
    if CFG["max_clips"]:   # smoke test: never overwrite the real outputs
        clip_paths, clip_ids, labels = (x[: CFG["max_clips"]] for x in (clip_paths, clip_ids, labels))
        out_dir = out_dir.with_name(out_dir.name + "_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs: dict[str, pd.DataFrame] = {}
    for condition in ("R", "C", "A"):
        preds = run_condition(model, clip_paths, clip_ids, labels, processor, CFG, condition)
        dfs[condition] = save_csv(preds, labels, id2label, out_dir / f"per_class_accuracy_UMT_ssv2_{condition}.csv")
    merge_conditions(dfs).to_csv(out_dir / "comparison.csv", index=False)
    print(f"\nComparison saved: {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
