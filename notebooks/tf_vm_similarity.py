"""
TF vs VideoMAE per-class accuracy similarity.

Finds classes where the two models' unperturbed accuracy is closest —
candidates where a temporal-vs-textural difference isn't just "one model
is bad at this class". SL status (temporal/static/unlabelled) carried
through from the VM file's sl_label column.

Outputs (outputs/stage1_class_selection_TF_VM/):
  TF_VM_similarity.csv   — class_name, delta, vm_accuracy, tf_accuracy, sl_status

Usage:
    uv run python notebooks/tf_vm_similarity.py
"""

from pathlib import Path
import pandas as pd

ROOT    = Path(__file__).parent.parent
TF_PATH = ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv"
VM_PATH = ROOT / "outputs/stage1_class_selection_VM/per_class_accuracy_VM_all.csv"
OUT_DIR = ROOT / "outputs/stage1_class_selection_TF_VM"
OUT_PATH = OUT_DIR / "TF_VM_similarity.csv"

tf = pd.read_csv(TF_PATH)[["class_id", "template", "accuracy", "total"]] \
    .rename(columns={"accuracy": "tf_accuracy", "total": "num_clips"})
vm = pd.read_csv(VM_PATH)[["class_id", "sl_label", "accuracy_R"]] \
    .rename(columns={"accuracy_R": "vm_accuracy"})

df = tf.merge(vm, on="class_id", how="inner")
df["delta"] = (df["vm_accuracy"] - df["tf_accuracy"]).abs()
df["sl_status"] = df["sl_label"] != "unlabelled"
df = df.rename(columns={"template": "class_name"})
df = df[["class_id", "class_name", "delta", "vm_accuracy", "tf_accuracy", "sl_status", "num_clips"]] \
    .sort_values("delta")

OUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"{len(df)} classes → {OUT_PATH}")
print(df.head(10).to_string(index=False))
