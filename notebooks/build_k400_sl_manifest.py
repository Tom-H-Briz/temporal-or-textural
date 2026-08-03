"""
Build outputs/Laura_SL/k400_manifest_SL_subset.json — K400 val clips whose class
maps to one of the 64 SL-matched classes in k400_sl_class_mapping.csv.

Mirrors LSL_subset.py's manifest_SL_subset.json for SSv2: whole validation set,
class-membership filter only, no held-out split, no correctness gate (per 03/08
CC brief — SSv2 has never excluded SAE-training clips from analysis, so K400
matches that behaviour instead of diverging from it).

Output schema mirrors manifest_SL_subset.json: {"temporal": [...], "static": [...]},
each entry {"id": <clip stem>, "label": <K400 class name>} — "label" stands in for
SSv2's "template" field since K400 clips carry a class name, not a template string.

Usage:
    uv run python notebooks/build_k400_sl_manifest.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).parent.parent
SL_DIR      = ROOT / "outputs" / "Laura_SL"
VAL_CSV     = ROOT / "data" / "kinetics400" / "annotations" / "val.csv"
MAPPING_CSV = SL_DIR / "k400_sl_class_mapping.csv"
OUT_JSON    = SL_DIR / "k400_manifest_SL_subset.json"


def load_sl_categories() -> dict[str, str]:
    df = pd.read_csv(MAPPING_CSV).dropna(subset=["matched_model_class_id"])
    return {row["matched_csv_label"]: row["sl_category"] for _, row in df.iterrows()}


def build_manifest(sl_by_label: dict[str, str]) -> dict:
    df = pd.read_csv(VAL_CSV)
    out = {"temporal": [], "static": []}
    for _, row in df.iterrows():
        category = sl_by_label.get(row["label"])
        if category is None:
            continue
        stem = f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}"
        out[category].append({"id": stem, "label": row["label"]})
    return out


def main() -> None:
    sl_by_label = load_sl_categories()
    print(f"  SL classes: {len(sl_by_label)}")

    manifest = build_manifest(sl_by_label)
    print(f"  Manifest clips — temporal: {len(manifest['temporal'])}  "
          f"static: {len(manifest['static'])}")
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved → {OUT_JSON}")

    labels = [e["label"] for e in manifest["temporal"] + manifest["static"]]
    counts = pd.Series(labels).value_counts()
    print(f"\n  Per-class counts — classes: {len(counts)}  min: {counts.min()}  max: {counts.max()}")
    thin = counts[counts < 10]
    if len(thin):
        print(f"  Thin classes (<10 clips): {thin.to_dict()}")


if __name__ == "__main__":
    main()
