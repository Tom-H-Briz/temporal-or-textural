"""
SL subset accuracy comparison — TF vs VideoMAE across perturbation conditions.

Outputs (outputs/analysis/sl_model_comparison/):
  sl_accuracy_comparison.csv   — summary rows at top, full per-class below
  sl_accuracy_comparison.png   — bar chart: TF vs VM across R/A/B/C by SL group
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent.parent
SL_PATH = ROOT / "outputs" / "Laura_SL" / "accuracy_SL_subset.csv"
TF_DIR  = ROOT / "outputs" / "stage1_class_selection_TF"
VM_DIR  = ROOT / "outputs" / "stage1_class_selection_VM"
OUT_DIR = ROOT / "outputs" / "analysis" / "sl_model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    ("TF", "R"):  TF_DIR / "per_class_accuracy_TF.csv",
    ("TF", "A"):  TF_DIR / "per_class_accuracy_TF_A.csv",
    ("TF", "B"):  TF_DIR / "per_class_accuracy_TF_B.csv",
    ("TF", "C"):  TF_DIR / "per_class_accuracy_TF_C.csv",
    ("VM", "R"):  ROOT / "outputs" / "stage1_class_selection" / "per_class_accuracy.csv",
    ("VM", "A"):  VM_DIR / "per_class_accuracy_VM_A.csv",
    ("VM", "B"):  VM_DIR / "per_class_accuracy_VM_B.csv",
    ("VM", "C"):  VM_DIR / "per_class_accuracy_VM_C.csv",
    ("VM", "C1"): VM_DIR / "per_class_accuracy_VM_C1.csv",
    ("VM", "C2"): VM_DIR / "per_class_accuracy_VM_C2.csv",
}

# ---------------------------------------------------------------------------
# Load SL subset — restrict to these class IDs throughout
# ---------------------------------------------------------------------------

sl = pd.read_csv(SL_PATH)[["class_id", "category", "template"]].rename(
    columns={"category": "sl_label"}
)
sl_ids = set(sl["class_id"])

# ---------------------------------------------------------------------------
# Build per-class combined table
# ---------------------------------------------------------------------------

per_class = sl.copy()
summary_records = []

for (model, cond), path in SOURCES.items():
    df = pd.read_csv(path)[["class_id", "correct", "total", "accuracy"]]
    df = df[df["class_id"].isin(sl_ids)].rename(columns={
        "correct":  f"correct_{model}_{cond}",
        "total":    f"total_{model}_{cond}",
        "accuracy": f"acc_{model}_{cond}",
    })
    per_class = per_class.merge(df, on="class_id", how="left")

    # Summary stats
    for group in ["temporal", "static"]:
        mask = per_class["sl_label"] == group
        c = per_class.loc[mask, f"correct_{model}_{cond}"].sum()
        t = per_class.loc[mask, f"total_{model}_{cond}"].sum()
        summary_records.append({
            "model": model, "condition": cond, "sl_group": group,
            "accuracy": round(c / t, 4) if t > 0 else float("nan"),
            "n_clips": int(t),
        })
    c_all = per_class[f"correct_{model}_{cond}"].sum()
    t_all = per_class[f"total_{model}_{cond}"].sum()
    summary_records.append({
        "model": model, "condition": cond, "sl_group": "all_SL",
        "accuracy": round(c_all / t_all, 4) if t_all > 0 else float("nan"),
        "n_clips": int(t_all),
    })

summary = pd.DataFrame(summary_records).sort_values(["condition", "model", "sl_group"])

# ---------------------------------------------------------------------------
# Write CSV: summary at top, blank row, per-class below
# ---------------------------------------------------------------------------

out_csv = OUT_DIR / "sl_accuracy_comparison.csv"
with open(out_csv, "w") as f:
    f.write("# SUMMARY\n")
    summary.to_csv(f, index=False)
    f.write("\n# PER CLASS\n")
    per_class.to_csv(f, index=False)
print(f"CSV → {out_csv}")

# ---------------------------------------------------------------------------
# Bar chart: TF vs VM for R/A/B/C, panels for temporal and static
# ---------------------------------------------------------------------------

shared_conds = ["R", "A", "B", "C"]
groups       = ["temporal", "static"]
models       = ["TF", "VM"]
colours      = {"TF": "steelblue", "VM": "darkorange"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, group in zip(axes, groups):
    x     = range(len(shared_conds))
    width = 0.35
    for i, model in enumerate(models):
        vals = []
        for cond in shared_conds:
            row = summary[(summary["model"] == model) &
                          (summary["condition"] == cond) &
                          (summary["sl_group"] == group)]
            vals.append(row["accuracy"].values[0] if len(row) else float("nan"))
        bars = ax.bar([xi + (i - 0.5) * width for xi in x], vals,
                      width=width, label=model, color=colours[model], alpha=0.85)
        for bar, v in zip(bars, vals):
            if not pd.isna(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(shared_conds)
    ax.set_title(f"{group.capitalize()} classes")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend()

fig.suptitle("TF vs VideoMAE — SL subset accuracy under perturbation", fontsize=12)
fig.tight_layout()
out_png = OUT_DIR / "sl_accuracy_comparison.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Chart → {out_png}")
