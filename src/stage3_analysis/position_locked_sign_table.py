"""
Position-locked sign table — sign_R and mean_s_R per class for selected features.

Reads per-class feature ranking CSVs. No recomputation.

Output: outputs/analysis/position_locked_sign_table/sign_table_vm_c1.md
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent.parent

BACKBONE  = "vm"
CONDITION = "c1"

FEATURES = [3513, 3558, 5578, 1990]

CLASSES = [
    (6,   "temporal"),
    (30,  "temporal"),
    (164, "temporal"),
    (171, "temporal"),
    (59,  "static"),
    (169, "static"),
    (173, "static"),
]

IN_DIR  = ROOT / "outputs" / "analysis" / f"per_class_feature_delta_{BACKBONE}_{CONDITION}"
OUT_DIR = ROOT / "outputs" / "analysis" / "position_locked_sign_table"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_row(class_id: int, feature_idx: int) -> dict:
    path = IN_DIR / f"class_{class_id}_feature_ranking.csv"
    df   = pd.read_csv(path, usecols=["feature_idx", "sign_R", "mean_s_R"])
    row  = df[df["feature_idx"] == feature_idx]
    if row.empty:
        return {"sign_R": float("nan"), "mean_s_R": float("nan")}
    return {"sign_R": int(row["sign_R"].iloc[0]), "mean_s_R": float(row["mean_s_R"].iloc[0])}


def sign_consistency(rows: list[dict]) -> str:
    signs = [r["sign_R"] for r in rows if not np.isnan(r["sign_R"])]
    if not signs:
        return "n/a"
    modal = max(set(signs), key=signs.count)
    return f"{signs.count(modal)} / {len(signs)}"


def make_table(feature_idx: int) -> str:
    temporal_rows, static_rows = [], []
    lines = [
        f"## Feature {feature_idx}",
        "",
        "| class | sl_label | sign_R | mean_s_R |",
        "|-------|----------|--------|----------|",
    ]

    for class_id, sl_label in CLASSES:
        row = load_feature_row(class_id, feature_idx)
        sign_str = f"{row['sign_R']:+d}" if not np.isnan(row["sign_R"]) else "n/a"
        mean_str = f"{row['mean_s_R']:.4f}" if not np.isnan(row["mean_s_R"]) else "n/a"
        lines.append(f"| {class_id} | {sl_label} | {sign_str} | {mean_str} |")
        if sl_label == "temporal":
            temporal_rows.append(row)
        else:
            static_rows.append(row)

    lines.append("")
    lines.append(f"sign_R consistency within temporal: {sign_consistency(temporal_rows)}")
    lines.append(f"sign_R consistency within static:   {sign_consistency(static_rows)}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_path = OUT_DIR / f"sign_table_{BACKBONE}_{CONDITION}.md"
    with open(out_path, "w") as f:
        f.write(f"# Sign table — {BACKBONE.upper()} {CONDITION.upper()}\n\n")
        for feat in FEATURES:
            f.write(make_table(feat))
            f.write("\n---\n\n")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
