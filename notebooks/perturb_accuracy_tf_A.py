"""Re-run TF condition A only — overwrites per_class_accuracy_TF_A.csv."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "stage1_dataset"))
sys.path.insert(0, str(Path(__file__).parent))

from perturb_accuracy_tf import CFG, PerturbedSSv2Dataset, run_condition, save_csv
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, _strip_brackets, load_metadata

label_map, clips, id2template = load_metadata(CFG["labels_path"], CFG["validation_path"])
video_dir = Path(CFG["video_dir"])

clip_paths, clip_ids, labels = [], [], []
for c in clips:
    template = _strip_brackets(c["template"])
    if template not in label_map:
        continue
    path = video_dir / f"{c['id']}.webm"
    if not path.exists():
        continue
    clip_paths.append(path)
    clip_ids.append(c["id"])
    labels.append(label_map[template])
print(f"  {len(clip_paths):,} clips")

model_cfg  = MODEL_REGISTRY[CFG["model_name"]]
checkpoint = CHECKPOINT_REGISTRY[(CFG["model_name"], "ssv2")]
processor  = model_cfg["processor_class"].from_pretrained(checkpoint)
model      = model_cfg["model_class"].from_pretrained(checkpoint)
model.to(CFG["device"]).eval()

out_dir = Path(CFG["output_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
preds = run_condition(model, clip_paths, clip_ids, labels, processor, CFG, "A")
save_csv(preds, labels, id2template, out_dir / "per_class_accuracy_TF_A.csv")
