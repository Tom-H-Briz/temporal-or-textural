"""
Scaffold feature mass contribution (%) — R / C1 / A.

Reads a dfa_mass_delta_vm_c1*.parquet for one specific SAE config. No DFA engine
calls — pure parquet analysis. Feature indices are only meaningful within the SAE
that produced them — never reuse SCAFFOLD_FEATURES across a different --layer/
--job-label/--sae-k without re-deriving the candidate list for that config.

Outputs (outputs/analysis/scaffold_mass_pct/):
    scaffold_mass_long_{suffix}.parquet               — per clip × condition × feature (floor)
    scaffold_pct_per_clip_{suffix}.parquet            — per clip × condition
    scaffold_pct_per_class_per_feature_{suffix}.csv
    scaffold_pct_per_slgroup_per_feature_{suffix}.csv
    scaffold_pct_combined_{suffix}.csv

Usage:
    uv run python src/stage3_analysis/scaffold_mass_pct.py --layer 7 --job-label 64 \\
        --features 1842,5578,1996,3513,1990,3558,4061,5552
    uv run python src/stage3_analysis/scaffold_mass_pct.py --layer 5 --job-label 64 \\
        --features 1394,1784,1919,2468,2577,3246,3325,6006
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

CFG = {
    "source_dir": ROOT / "outputs/analysis/dfa_mass_delta_vm_c1",
    "out_dir":    ROOT / "outputs/analysis/scaffold_mass_pct",
}

CONDITIONS = ["R", "C1", "A"]


def locate_source(layer: int, job_label: str, sae_k: int) -> Path:
    """Prefer the layer/job/k-suffixed file; fall back to the original unsuffixed
    L7/job64 baseline file if this is that exact config and no suffixed copy exists."""
    suffixed = CFG["source_dir"] / f"dfa_mass_delta_vm_c1_l{layer}_job{job_label}_k{sae_k}.parquet"
    if suffixed.exists():
        return suffixed
    legacy = CFG["source_dir"] / "dfa_mass_delta_vm_c1.parquet"
    if (layer, job_label, sae_k) == (7, "64", 64) and legacy.exists():
        return legacy
    raise FileNotFoundError(f"No source parquet found: {suffixed} (or legacy {legacy})")


def extract_long(df: pd.DataFrame, scaffold_features: list[int]) -> pd.DataFrame:
    """Stage 1 — long format: one row per (clip, condition, feature)."""
    meta = df[["clip_id", "class_id", "sl_label", "correct_C1", "correct_A"]].reset_index(drop=True)
    chunks = []
    for cond in CONDITIONS:
        mat = np.stack(df[f"signed_vec_{cond}"].to_numpy()).astype(np.float32)  # (N, dict_size)
        total = np.abs(mat).sum(axis=1)  # (N,)
        for feat in scaffold_features:
            sv = mat[:, feat]
            chunk = meta.copy()
            chunk["condition"] = cond
            chunk["feature_idx"] = feat
            chunk["signed_val"] = sv.astype(np.float32)
            chunk["abs_val"] = np.abs(sv).astype(np.float32)
            chunk["total_abs_signed"] = total.astype(np.float32)
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def compute_pct_per_clip(long: pd.DataFrame) -> pd.DataFrame:
    """Stage 2 — per-clip scaffold %: sum abs_val across features, divide by total."""
    grp_cols = ["clip_id", "class_id", "sl_label", "condition", "correct_C1", "correct_A"]
    agg = long.groupby(grp_cols, as_index=False).agg(
        scaffold_abs_sum=("abs_val", "sum"),
        total_abs_signed=("total_abs_signed", "first"),
    )
    agg["scaffold_pct"] = agg["scaffold_abs_sum"] / agg["total_abs_signed"]
    return agg


def _add_per_clip_ratio(long: pd.DataFrame) -> pd.DataFrame:
    out = long.copy()
    out["pct"] = out["abs_val"] / out["total_abs_signed"]
    return out


def agg_per_class_per_feature(long: pd.DataFrame) -> pd.DataFrame:
    """Stage 3a — mean per-clip ratio, per (class, condition, feature)."""
    df = _add_per_clip_ratio(long)
    grp = df.groupby(["class_id", "sl_label", "condition", "feature_idx"], as_index=False)
    return grp["pct"].agg(mean_pct="mean", n_clips="count")


def agg_per_slgroup_per_feature(long: pd.DataFrame) -> pd.DataFrame:
    """Stage 3b — mean per-clip ratio, per (sl_label, condition, feature), pooled across classes."""
    df = _add_per_clip_ratio(long)
    grp = df.groupby(["sl_label", "condition", "feature_idx"], as_index=False)
    return grp["pct"].agg(mean_pct="mean", n_clips="count")


def agg_combined(pct_per_clip: pd.DataFrame) -> pd.DataFrame:
    """Stage 3c — combined scaffold mean_pct at class / sl_group / overall level."""
    rows = []
    for cond in CONDITIONS:
        sub = pct_per_clip[pct_per_clip["condition"] == cond]
        for (class_id, sl), g in sub.groupby(["class_id", "sl_label"]):
            rows.append({"grouping_level": "class", "grouping_value": str(class_id),
                         "condition": cond, "mean_pct": g["scaffold_pct"].mean(), "n_clips": len(g)})
        for sl, g in sub.groupby("sl_label"):
            rows.append({"grouping_level": "sl_group", "grouping_value": sl,
                         "condition": cond, "mean_pct": g["scaffold_pct"].mean(), "n_clips": len(g)})
        rows.append({"grouping_level": "overall", "grouping_value": "all",
                     "condition": cond, "mean_pct": sub["scaffold_pct"].mean(), "n_clips": len(sub)})
    return pd.DataFrame(rows)


def sanity_check(df_a: pd.DataFrame, df_b: pd.DataFrame, df_c: pd.DataFrame) -> None:
    """Assert sum of per-feature mean_pcts == combined mean_pct, by linearity of per-clip mean."""
    tol = 1e-5

    # Class level: sum over features in table a == table c per (class_id, condition)
    a_sums = df_a.groupby(["class_id", "condition"])["mean_pct"].sum().reset_index(name="a_sum")
    c_class = df_c[df_c["grouping_level"] == "class"].copy()
    c_class["class_id"] = c_class["grouping_value"].astype(int)
    merged_a = a_sums.merge(c_class[["class_id", "condition", "mean_pct"]], on=["class_id", "condition"])
    bad_a = merged_a[(merged_a["a_sum"] - merged_a["mean_pct"]).abs() > tol]
    assert len(bad_a) == 0, f"Sanity FAIL (class-level, table a vs c):\n{bad_a}"

    # SL-group level: sum over features in table b == table c per (sl_label, condition)
    b_sums = df_b.groupby(["sl_label", "condition"])["mean_pct"].sum().reset_index(name="b_sum")
    c_grp = df_c[df_c["grouping_level"] == "sl_group"].rename(columns={"grouping_value": "sl_label"})
    merged_b = b_sums.merge(c_grp[["sl_label", "condition", "mean_pct"]], on=["sl_label", "condition"])
    bad_b = merged_b[(merged_b["b_sum"] - merged_b["mean_pct"]).abs() > tol]
    assert len(bad_b) == 0, f"Sanity FAIL (sl_group-level, table b vs c):\n{bad_b}"

    print("  Sanity check passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--job-label", type=str, default="64")
    parser.add_argument("--sae-k", type=int, default=64)
    parser.add_argument("--features", type=str, required=True,
                         help="comma-separated candidate feature indices for this SAE config")
    args = parser.parse_args()
    scaffold_features = [int(f) for f in args.features.split(",")]
    out_suffix = f"l{args.layer}_job{args.job_label}_k{args.sae_k}"

    out_dir: Path = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    src = locate_source(args.layer, args.job_label, args.sae_k)
    print(f"Source: {src}")
    print(f"Candidate features ({len(scaffold_features)}): {scaffold_features}")
    df = pd.read_parquet(src)
    print(f"  {len(df):,} clips, {df['class_id'].nunique()} classes, {df['sl_label'].value_counts().to_dict()}")

    print("Stage 1: extracting long format...")
    long = extract_long(df, scaffold_features)
    long.to_parquet(out_dir / f"scaffold_mass_long_{out_suffix}.parquet", index=False)
    print(f"  {len(long):,} rows → scaffold_mass_long_{out_suffix}.parquet")

    print("Stage 2: per-clip scaffold %...")
    pct_per_clip = compute_pct_per_clip(long)
    pct_per_clip.to_parquet(out_dir / f"scaffold_pct_per_clip_{out_suffix}.parquet", index=False)
    print(f"  {len(pct_per_clip):,} rows → scaffold_pct_per_clip_{out_suffix}.parquet")

    print("Stage 3: aggregations...")
    df_a = agg_per_class_per_feature(long)
    df_b = agg_per_slgroup_per_feature(long)
    df_c = agg_combined(pct_per_clip)

    sanity_check(df_a, df_b, df_c)

    df_a.to_csv(out_dir / f"scaffold_pct_per_class_per_feature_{out_suffix}.csv", index=False)
    df_b.to_csv(out_dir / f"scaffold_pct_per_slgroup_per_feature_{out_suffix}.csv", index=False)
    df_c.to_csv(out_dir / f"scaffold_pct_combined_{out_suffix}.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
