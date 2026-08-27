"""
Ahn-32 flip rate — L7 and L5+L7 cross-layer, completing what
ablation_summary_ahn32.py only did for L5.

L7 reuses that exact same technique (inner-join clips onto the Ahn mapping,
overwrite sl_label with ahn_label, then ablation_summary.py's compute_flip_rate
— nothing reimplemented). L5+L7 cross-layer needs its own path:
ablation_cross_l5_l7.parquet has no ablation_target/perturbation_condition
columns (single R-only joint target by design, see that script's docstring),
so compute_flip_rate's groupby doesn't apply — flip rate is computed directly
per ahn_label instead, same "1 - correct_ablated.mean()" definition.

No new extraction or GPU compute — both read already-computed parquets.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_flip_rate_l7_job7ep_k64_ahn32.csv
    ablation_flip_rate_l5_l7_cross_ahn32.csv

Usage:
    uv run python src/stage3_analysis/ahn32_flip_rate_l7_and_cross.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_summary import compute_flip_rate

CFG = {
    "l7_parquet":    ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l7_job7ep_k64.parquet",
    "cross_parquet": ROOT / "outputs/analysis/scaffold_ablation/ablation_cross_l5_l7.parquet",
    "mapping_csv":   ROOT / "outputs/analysis/taxonomy/sl32_vs_ahn_mapping.csv",
    "out_dir":       ROOT / "outputs/analysis/scaffold_ablation",
}


def load_ahn_relabeled(parquet_path: Path, mapping_csv: Path) -> pd.DataFrame:
    """Same technique as ablation_summary_ahn32.py's load_ahn_relabeled —
    inner join restricts to the 32 Ahn-mapped classes, sl_label is
    overwritten with ahn_label so downstream functions expecting sl_label
    need no changes."""
    df = pd.read_parquet(parquet_path)
    mapping = pd.read_csv(mapping_csv)[["ssv2_class_id", "ahn_label"]]
    merged = df.merge(mapping, left_on="class_id", right_on="ssv2_class_id", how="inner")
    merged["sl_label"] = merged["ahn_label"]
    return merged


def cross_layer_flip_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Same "1 - correct_ablated.mean()" definition as compute_flip_rate, but
    grouped by sl_label (ahn_label) alone — no ablation_target/condition axes
    exist in this file (single R-only joint L5+L7 target by design)."""
    flip = lambda s: 1 - s.mean()
    by_label = df.groupby("sl_label", as_index=False).agg(
        flip_rate=("correct_ablated", flip), n_clips=("correct_ablated", "count"))
    overall = pd.DataFrame([{
        "sl_label": "overall", "flip_rate": flip(df["correct_ablated"]), "n_clips": len(df),
    }])
    return pd.concat([by_label, overall], ignore_index=True)


def main() -> None:
    l7 = load_ahn_relabeled(CFG["l7_parquet"], CFG["mapping_csv"])
    l7_flip = compute_flip_rate(l7)
    l7_flip.to_csv(CFG["out_dir"] / "ablation_flip_rate_l7_job7ep_k64_ahn32.csv", index=False)
    print("=== L7 Ahn-32 flip rate ===")
    print(l7_flip.to_string(index=False))

    cross = load_ahn_relabeled(CFG["cross_parquet"], CFG["mapping_csv"])
    cross_flip = cross_layer_flip_rate(cross)
    cross_flip.to_csv(CFG["out_dir"] / "ablation_flip_rate_l5_l7_cross_ahn32.csv", index=False)
    print("\n=== L5+L7 cross-layer Ahn-32 flip rate (R only) ===")
    print(cross_flip.to_string(index=False))


if __name__ == "__main__":
    main()
