"""
VideoMAE perturbation accuracy summary — one-stop shop.

Combines R (unperturbed) with VM conditions A/B/C/C1/C2.
SL classes labelled as temporal/static where available.

Outputs (outputs/stage1_class_selection_VM/):
  per_class_accuracy_VM_all.csv   — per-class accuracy + SL label for all conditions
  summary_statistics_VM.csv       — overall + per-SL-group accuracy per condition
  overall_accuracy_VM_all_conditions.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent.parent
VM_DIR  = ROOT / "outputs" / "stage1_class_selection_VM"
R_PATH  = ROOT / "outputs" / "stage1_class_selection" / "per_class_accuracy.csv"
SL_PATH = ROOT / "outputs" / "Laura_SL" / "accuracy_SL_subset.csv"

CONDITIONS = {
    "R":  R_PATH,
    "A":  VM_DIR / "per_class_accuracy_VM_A.csv",
    "B":  VM_DIR / "per_class_accuracy_VM_B.csv",
    "C":  VM_DIR / "per_class_accuracy_VM_C.csv",
    "C1": VM_DIR / "per_class_accuracy_VM_C1.csv",
    "C2": VM_DIR / "per_class_accuracy_VM_C2.csv",
}

# ---------------------------------------------------------------------------
# Load SL labels
# ---------------------------------------------------------------------------

sl_labels = pd.read_csv(SL_PATH)[["class_id", "category"]].rename(columns={"category": "sl_label"})

# ---------------------------------------------------------------------------
# Load and combine per-class accuracy
# ---------------------------------------------------------------------------

base = pd.read_csv(R_PATH)[["class_id", "template"]].copy()
combined = base.merge(sl_labels, on="class_id", how="left")
combined["sl_label"] = combined["sl_label"].fillna("unlabelled")

overall = {}
for cond, path in CONDITIONS.items():
    df = pd.read_csv(path)[["class_id", "correct", "total", "accuracy"]]
    df = df.rename(columns={"correct": f"correct_{cond}",
                             "total":   f"total_{cond}",
                             "accuracy": f"accuracy_{cond}"})
    combined = combined.merge(df, on="class_id", how="left")
    overall[cond] = combined[f"correct_{cond}"].sum() / combined[f"total_{cond}"].sum()

combined.to_csv(VM_DIR / "per_class_accuracy_VM_all.csv", index=False)
print(f"Per-class CSV → {VM_DIR / 'per_class_accuracy_VM_all.csv'}")

# ---------------------------------------------------------------------------
# Summary statistics: overall + per SL group
# ---------------------------------------------------------------------------

summary_rows = []
for cond in CONDITIONS:
    row = {"condition": cond, "overall": overall[cond]}
    for group in ["temporal", "static", "unlabelled"]:
        mask = combined["sl_label"] == group
        c = combined.loc[mask, f"correct_{cond}"].sum()
        t = combined.loc[mask, f"total_{cond}"].sum()
        row[group] = c / t if t > 0 else float("nan")
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(VM_DIR / "summary_statistics_VM.csv", index=False)

print("\nOverall top-1 accuracy:")
for _, row in summary.iterrows():
    print(f"  {row['condition']:3s}  overall={row['overall']:.4f}  "
          f"temporal={row['temporal']:.4f}  static={row['static']:.4f}")

# ---------------------------------------------------------------------------
# Bar chart — overall + SL groups per condition
# ---------------------------------------------------------------------------

conds   = list(CONDITIONS.keys())
x       = range(len(conds))
width   = 0.25
colours = {"overall": "steelblue", "temporal": "seagreen", "static": "darkorange"}

fig, ax = plt.subplots(figsize=(11, 5))
for i, (key, colour) in enumerate(colours.items()):
    vals  = [summary.loc[summary["condition"] == c, key].values[0] for c in conds]
    bars  = ax.bar([xi + (i - 1) * width for xi in x], vals,
                   width=width, label=key.capitalize(), color=colour, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

ax.set_xticks(list(x))
ax.set_xticklabels(conds)
ax.set_ylim(0, max(overall.values()) * 1.18)
ax.set_xlabel("Condition")
ax.set_ylabel("Top-1 accuracy")
ax.set_title("VideoMAE — accuracy under perturbation (overall / temporal / static)")
ax.legend()
fig.tight_layout()
out = VM_DIR / "overall_accuracy_VM_all_conditions.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"\nBar chart → {out}")
