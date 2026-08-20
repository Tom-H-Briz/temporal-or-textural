"""
L5 VM SSv2 ablation impact, regrouped by Ahn's temporal/static split instead
of Sevilla-Lara's — same statistics as ablation_summary.py (reused directly,
not reimplemented), just with sl_label overwritten by ahn_label and restricted
to the 32-class subset ahn_taxonomy_mapping.py actually mapped (the other 3
of run_ablation.py's 35 SL-pilot classes have no Ahn correspondence yet).

Usage:
    uv run python src/stage3_analysis/ablation_summary_ahn32.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_targets import get_targets, group_targets, singleton_targets
from stage3_analysis.ablation_summary import (
    compute_summary, compute_diff_in_diff, compute_additivity, compute_flip_rate,
)

CFG = {
    "results_parquet": ROOT / "outputs/analysis/scaffold_ablation/ablation_results_long_l5_job7ep_k64.parquet",
    "mapping_csv": ROOT / "outputs/analysis/taxonomy/sl32_vs_ahn_mapping.csv",
    "out_dir": ROOT / "outputs/analysis/scaffold_ablation",
}


def load_ahn_relabeled(cfg: dict) -> pd.DataFrame:
    """Inner-join restricts to the 32 classes ahn_taxonomy_mapping.py mapped
    (drops the 3 extra SL-pilot classes with no Ahn correspondence). Rows
    whose ahn_label is 'unassigned' are kept — compute_diff_in_diff only
    ever pulls the 'temporal'/'static' rows, so they fall out naturally
    rather than needing an explicit drop."""
    df = pd.read_parquet(cfg["results_parquet"])
    mapping = pd.read_csv(cfg["mapping_csv"])[["ssv2_class_id", "ahn_label"]]
    merged = df.merge(mapping, left_on="class_id", right_on="ssv2_class_id", how="inner")
    merged["sl_label"] = merged["ahn_label"]  # reuse compute_* as-is, they read "sl_label"
    return merged


def main() -> None:
    cfg = CFG
    df = load_ahn_relabeled(cfg)
    print(f"  {len(df):,} rows  |  {df['class_id'].nunique()} classes  |  "
          f"ahn_label counts: {df.groupby('class_id')['ahn_label'].first().value_counts().to_dict()}")

    targets = get_targets("ssv2", 5)
    singleton, group_names = singleton_targets(targets), group_targets(targets)

    summary = compute_summary(df)
    did = compute_diff_in_diff(summary)
    add = compute_additivity(summary, singleton, group_names)
    flip = compute_flip_rate(df)
    flip_did = compute_diff_in_diff(flip, value_col="flip_rate")

    out_dir: Path = cfg["out_dir"]
    summary.to_csv(out_dir / "ablation_summary_l5_job7ep_k64_ahn32.csv", index=False)
    did.to_csv(out_dir / "ablation_diff_in_diff_l5_job7ep_k64_ahn32.csv", index=False)
    add.to_csv(out_dir / "ablation_additivity_l5_job7ep_k64_ahn32.csv", index=False)
    flip.to_csv(out_dir / "ablation_flip_rate_l5_job7ep_k64_ahn32.csv", index=False)
    flip_did.to_csv(out_dir / "ablation_flip_rate_diff_in_diff_l5_job7ep_k64_ahn32.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(did.to_string(index=False))
    print()
    print(add.to_string(index=False))
    print()
    print(flip.to_string(index=False))
    print()
    print(flip_did.to_string(index=False))


if __name__ == "__main__":
    main()
