"""
Extract Laura SL temporal/static class subsets, build a filtered JSON manifest,
and pull per-class TF accuracy for those classes.

Outputs (both to outputs/Laura_SL/):
  - manifest_SL_subset.json  — val clips belonging to SL temporal or static classes
  - accuracy_SL_subset.csv   — per-class TF accuracy for those classes
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT    = Path(__file__).parent.parent
SL_DIR  = ROOT / "outputs" / "Laura_SL"
VAL_JSON    = ROOT / "data" / "ssv2" / "labels" / "validation.json"
ACC_CSV     = ROOT / "outputs" / "stage1_class_selection_TF" / "per_class_accuracy_TF.csv"
ST_CSV      = SL_DIR / "Static_Temporal.csv"
OUT_JSON    = SL_DIR / "manifest_SL_subset.json"
OUT_ACC_CSV = SL_DIR / "accuracy_SL_subset.csv"


def load_class_lists() -> tuple[set[str], set[str]]:
    df = pd.read_csv(ST_CSV, encoding="utf-8-sig", header=0)
    temporal = set(df.iloc[:, 0].dropna().str.strip())
    static   = set(df.iloc[:, 1].dropna().str.strip())
    return temporal, static


def build_manifest(temporal: set[str], static: set[str]) -> dict:
    with open(VAL_JSON) as f:
        clips = json.load(f)

    def strip(t: str) -> str:
        return t.replace("[", "").replace("]", "")

    out = {"temporal": [], "static": []}
    for c in clips:
        t = strip(c["template"])
        if t in temporal:
            out["temporal"].append({"id": c["id"], "template": t})
        elif t in static:
            out["static"].append({"id": c["id"], "template": t})
    return out


def build_accuracy(temporal: set[str], static: set[str]) -> pd.DataFrame:
    df  = pd.read_csv(ACC_CSV)
    all_classes = temporal | static
    sub = df[df["template"].isin(all_classes)].copy()
    sub["category"] = sub["template"].apply(
        lambda t: "temporal" if t in temporal else "static"
    )
    return sub[["category", "class_id", "template", "correct", "total", "accuracy"]]


def main() -> None:
    temporal, static = load_class_lists()
    print(f"  Temporal classes: {len(temporal)}  Static classes: {len(static)}")

    manifest = build_manifest(temporal, static)
    print(f"  Manifest clips — temporal: {len(manifest['temporal'])}  "
          f"static: {len(manifest['static'])}")
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved → {OUT_JSON}")

    acc = build_accuracy(temporal, static)
    acc.to_csv(OUT_ACC_CSV, index=False)
    print(f"  Saved → {OUT_ACC_CSV}")
    print(f"\n  Mean accuracy — temporal: {acc[acc['category']=='temporal']['accuracy'].mean():.4f}"
          f"  static: {acc[acc['category']=='static']['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
