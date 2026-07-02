"""
Scaffold feature mass contribution (%) — R / C1 / A.

Reads dfa_mass_delta_vm_c1.parquet. No DFA engine calls — pure parquet analysis.

Outputs (outputs/analysis/scaffold_mass_pct/):
    scaffold_mass_long.parquet               — per clip × condition × feature (floor)
    scaffold_pct_per_clip.parquet            — per clip × condition
    scaffold_pct_per_class_per_feature.csv
    scaffold_pct_per_slgroup_per_feature.csv
    scaffold_pct_combined.csv

Do not run until SCAFFOLD_FEATURES is finalised.

Usage:
    uv run python src/stage3_analysis/scaffold_mass_pct.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

SCAFFOLD_FEATURES: list[int] = [1842, 5578, 1996, 3513, 1990, 3558, 4061, 5552]
# Tubelets:                        0     1     3     4     5     6     7     7
# 4061: soft lock — R share 0.974 (σ=0.013) vs 5552's 0.990 (σ=0.004); A share 0.864 vs 1.000.
# Kept in set; disambiguation deferred to ablation (isolate-4061 condition). Finalised 020726.

CFG = {
    "source_glob": "outputs/analysis/**/dfa_mass_delta_vm_c1.parquet",
    "out_dir":     ROOT / "outputs/analysis/scaffold_mass_pct",
}

CONDITIONS = ["R", "C1", "A"]


def locate_source() -> Path:
    matches = list(ROOT.glob(CFG["source_glob"]))
    assert len(matches) == 1, f"Expected exactly 1 source parquet, found: {matches}"
    return matches[0]


def extract_long(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 — long format: one row per (clip, condition, feature)."""
    meta = df[["clip_id", "class_id", "sl_label", "correct_C1", "correct_A"]].reset_index(drop=True)
    chunks = []
    for cond in CONDITIONS:
        mat = np.stack(df[f"signed_vec_{cond}"].to_numpy()).astype(np.float32)  # (N, 6144)
        total = np.abs(mat).sum(axis=1)  # (N,)
        for feat in SCAFFOLD_FEATURES:
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
    assert SCAFFOLD_FEATURES, (
        "SCAFFOLD_FEATURES is empty — do not run until the position-lock fix run "
        "confirms the final feature set (7 or 8 features, pending feature 4061 status)."
    )

    out_dir: Path = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    src = locate_source()
    print(f"Source: {src}")
    df = pd.read_parquet(src)
    print(f"  {len(df):,} clips, {df['class_id'].nunique()} classes, {df['sl_label'].value_counts().to_dict()}")

    print("Stage 1: extracting long format...")
    long = extract_long(df)
    long.to_parquet(out_dir / "scaffold_mass_long.parquet", index=False)
    print(f"  {len(long):,} rows → scaffold_mass_long.parquet")

    print("Stage 2: per-clip scaffold %...")
    pct_per_clip = compute_pct_per_clip(long)
    pct_per_clip.to_parquet(out_dir / "scaffold_pct_per_clip.parquet", index=False)
    print(f"  {len(pct_per_clip):,} rows → scaffold_pct_per_clip.parquet")

    print("Stage 3: aggregations...")
    df_a = agg_per_class_per_feature(long)
    df_b = agg_per_slgroup_per_feature(long)
    df_c = agg_combined(pct_per_clip)

    sanity_check(df_a, df_b, df_c)

    df_a.to_csv(out_dir / "scaffold_pct_per_class_per_feature.csv", index=False)
    df_b.to_csv(out_dir / "scaffold_pct_per_slgroup_per_feature.csv", index=False)
    df_c.to_csv(out_dir / "scaffold_pct_combined.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
