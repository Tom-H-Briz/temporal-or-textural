"""
Four-bucket (noise/sign_flip/decrease/increase) shuffle-disruption behaviour,
stratified by SL vs. Ahn taxonomy — CC brief 12/08/26, parent workbook_entry
_120826.md §5a.9. Re-aggregation only: no new DFA, ablation, or bucket
re-derivation. Reuses clip_shuffle_disruption.py's persisted per-clip bucket
fractions, ahn_taxonomy_mapping.py's class-correspondence table, and (as of
13/08) ssv2_{vm,tf}_full_detail.parquet's per-(feature,clip) instance data
as-is.

Correction (13/08): an earlier version of this script declared the
instance-weighted metric blocked, having only found class_feature_breakdown
.py's 4+4-class spot-check output during Step 0. outputs/analysis/shuffle_
reduction_composition/ssv2_{vm,tf}_full_detail.parquet was missed — it holds
the full per-(feature,clip) bucket data (VM: 44,910 rows/4,491 clips/33
classes; TF: 35,770/3,577/31) that instance_weighted_summary() and
top12_concentration() below use. Verified against a known figure: TF's top-12
sign-flip features here account for 82.546% of TF sign-flip instances,
matching the brief's cited "82.5% at L7" exactly.

Usage:
    uv run python src/stage3_analysis/feature_behave_SLvsASL.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

BUCKETS = ("noise", "sign_flip", "decrease", "increase")

CFG = {
    "vm_csv": ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_vm_clip_shuffle_disruption.csv",
    "tf_csv": ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_tf_clip_shuffle_disruption.csv",
    "vm_detail": ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_vm_full_detail.parquet",
    "tf_detail": ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_tf_full_detail.parquet",
    "top_n": 12,
    "sl_csv": ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv",
    "ahn_mapping_csv": ROOT / "outputs/analysis/taxonomy/sl32_vs_ahn_mapping.csv",
    "out_dir": ROOT / "outputs/analysis/taxonomy",
}


def load_backbone(name: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["backbone"] = name
    return df


def attach_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """sl_label: left-join, covers all classes (35-class SL-pilot source).
    ahn_label: left-join against the 32-class mapping only — classes outside
    it (e.g. VM's class 160) get NaN and are excluded from Ahn stratification
    downstream, not silently folded into a bucket."""
    sl = pd.read_csv(cfg["sl_csv"])[["class_id", "category"]].rename(columns={"category": "sl_label"})
    ahn = pd.read_csv(cfg["ahn_mapping_csv"])[["ssv2_class_id", "ahn_label"]]
    df = df.merge(sl, on="class_id", how="left")
    df = df.merge(ahn, left_on="class_id", right_on="ssv2_class_id", how="left")
    return df.drop(columns=["ssv2_class_id"])


def coverage_report(df: pd.DataFrame, backbone: str) -> list[str]:
    """Step 0/1 — row/class counts and per-stratum class counts, both
    taxonomies, so denominators are visible before any ratio is trusted."""
    lines = [f"### {backbone.upper()}", "",
             f"- {len(df):,} clips, {df['class_id'].nunique()} classes"]
    for tax_col in ("sl_label", "ahn_label"):
        counts = df.groupby("class_id")[tax_col].first().value_counts(dropna=False)
        lines.append(f"- `{tax_col}` classes per stratum: " +
                      ", ".join(f"{k}={v}" for k, v in counts.items()))
    lines.append("")
    return lines


def per_class_fracs(df: pd.DataFrame) -> pd.DataFrame:
    """Mean frac_* per class — 'how noisy is a typical clip in this class',
    collapsed to one row per class (class-weighted unit, not instance)."""
    agg = {f"frac_{b}": "mean" for b in BUCKETS}
    out = df.groupby(["backbone", "class_id"], as_index=False).agg(agg)
    labels = df.groupby("class_id")[["sl_label", "ahn_label"]].first()
    return out.merge(labels, on="class_id", how="left")


def class_weighted_summary(per_class: pd.DataFrame) -> pd.DataFrame:
    """Step 2B — median (primary) + mean across classes, per (taxonomy,
    backbone, stratum, bucket). Long format, one row per bucket."""
    rows = []
    for taxonomy, label_col in (("SL", "sl_label"), ("Ahn", "ahn_label")):
        grp = per_class.dropna(subset=[label_col]).groupby(["backbone", label_col])
        for (backbone, stratum), sub in grp:
            for b in BUCKETS:
                rows.append({
                    "taxonomy": taxonomy, "backbone": backbone, "stratum": stratum,
                    "bucket": b, "n_classes": len(sub),
                    "median_frac": float(sub[f"frac_{b}"].median()),
                    "mean_frac": float(sub[f"frac_{b}"].mean()),
                })
    return pd.DataFrame(rows)


def load_detail(name: str, path: Path, cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["backbone"] = name
    return attach_labels(df, cfg)


def instance_weighted_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Step 2A table 1 — per (taxonomy, backbone, stratum, bucket): instance
    count and share of that stratum's total instances. Unit = (feature, clip)
    row, not class — a class with 500 clips contributes 500x the rows of a
    class with 1."""
    rows = []
    for taxonomy, label_col in (("SL", "sl_label"), ("Ahn", "ahn_label")):
        sub_all = detail.dropna(subset=[label_col])
        for (backbone, stratum), sub in sub_all.groupby(["backbone", label_col]):
            total = len(sub)
            for b in BUCKETS:
                n = int((sub["bucket"] == b).sum())
                rows.append({"taxonomy": taxonomy, "backbone": backbone, "stratum": stratum,
                            "bucket": b, "instance_count": n, "share_of_stratum": n / total})
    return pd.DataFrame(rows)


def _top_n_share(sign_flip: pd.DataFrame, n: int) -> float:
    counts = sign_flip["feature_id"].value_counts()
    return float(counts.head(n).sum() / len(sign_flip)) if len(sign_flip) else float("nan")


def top12_concentration(detail: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Step 2A table 2 — cumulative share of sign_flip instances the top-12
    features carry, per (taxonomy, backbone, stratum + 'overall'). Instance-
    weighted (pooled across clips) AND a class-weighted analog (per-class
    top-12 share, then median/mean across classes) side by side — the literal
    comparison the brief's disagreement-flagging instruction needs."""
    n = cfg["top_n"]
    sf_all = detail[detail["bucket"] == "sign_flip"]
    rows = []
    for taxonomy, label_col in (("SL", "sl_label"), ("Ahn", "ahn_label")):
        sf = sf_all.dropna(subset=[label_col])
        for backbone, bsub in sf.groupby("backbone"):
            for stratum, ssub in list(bsub.groupby(label_col)) + [("overall", bsub)]:
                instance_wt = _top_n_share(ssub, n)
                per_class = ssub.groupby("class_id").apply(
                    lambda g: _top_n_share(g, n), include_groups=False)
                rows.append({
                    "taxonomy": taxonomy, "backbone": backbone, "stratum": stratum,
                    "n_sign_flip_instances": len(ssub), "n_classes": ssub["class_id"].nunique(),
                    "instance_weighted_top12_share": instance_wt,
                    "class_weighted_median_top12_share": float(per_class.median()) if len(per_class) else float("nan"),
                    "class_weighted_mean_top12_share": float(per_class.mean()) if len(per_class) else float("nan"),
                })
    return pd.DataFrame(rows)


def vm_tf_top12_ratio(concentration: pd.DataFrame) -> pd.DataFrame:
    """Step 3's second falsifier — VM/TF ratio of top-12 sign-flip coverage
    (instance-weighted), 'overall' stratum only (the figure the brief cites,
    82.5% at L7, is unstratified)."""
    overall = concentration[concentration["stratum"] == "overall"]
    rows = []
    for taxonomy, sub in overall.groupby("taxonomy"):
        sub = sub.set_index("backbone")
        if not {"vm", "tf"}.issubset(sub.index):
            continue
        rows.append({
            "taxonomy": taxonomy,
            "vm_top12_share": sub.loc["vm", "instance_weighted_top12_share"],
            "tf_top12_share": sub.loc["tf", "instance_weighted_top12_share"],
            "vm_tf_ratio": sub.loc["vm", "instance_weighted_top12_share"] / sub.loc["tf", "instance_weighted_top12_share"],
        })
    return pd.DataFrame(rows)


def _top12_verdict_line(top12_ratio: pd.DataFrame) -> str:
    """Same treatment as _verdict_line, for the newly-unblocked second
    falsifier — factual swing only, no confirmed/failed label invented here."""
    pivot = top12_ratio.set_index("taxonomy")["vm_tf_ratio"]
    swing = abs(pivot["Ahn"] - pivot["SL"])
    return (f"**Result: |Ahn − SL| swing in VM/TF top-12 coverage ratio = {swing:.4f}** "
            f"(SL: {pivot['SL']:.4f} → Ahn: {pivot['Ahn']:.4f}).")


def disagreement_flags(concentration: pd.DataFrame, threshold: float = 0.15) -> list[str]:
    """Per the brief: 'flag explicitly if instance-weighted and class-weighted
    disagree in direction... should not be buried in a table.' Flags rows
    where the two top-12-share estimates differ by more than `threshold`."""
    flags = []
    for _, r in concentration.iterrows():
        gap = r["instance_weighted_top12_share"] - r["class_weighted_median_top12_share"]
        if abs(gap) > threshold:
            direction = "carried by a few high-n classes" if gap > 0 else "diffuse pooled but concentrated per-class"
            flags.append(f"- **{r['taxonomy']}/{r['backbone']}/{r['stratum']}**: instance-weighted "
                         f"{r['instance_weighted_top12_share']:.3f} vs. class-weighted median "
                         f"{r['class_weighted_median_top12_share']:.3f} (gap {gap:+.3f}) — {direction}")
    return flags if flags else ["- None: no (taxonomy, backbone, stratum) exceeds the "
                                 f"{threshold} gap threshold between the two weightings."]


def df_to_md_table(df: pd.DataFrame) -> str:
    """Manual markdown table — avoids adding the `tabulate` dependency just
    for pandas.to_markdown(), which isn't installed in this project's venv."""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def _verdict_line(ratio: pd.DataFrame) -> str:
    """Factual observation only (Step 4: no mechanism interpretation) — the
    max absolute swing in the VM/TF median noise ratio between SL and Ahn,
    read against the pre-registered 'should be small' expectation. Temporal/
    static are the only strata present in both taxonomies (Ahn's 'unassigned'
    has no SL counterpart, so it's excluded from this specific comparison)."""
    comparable = ratio[ratio["stratum"].isin(["temporal", "static"])]
    pivot = comparable.pivot(index="stratum", columns="taxonomy", values="vm_tf_ratio_median")
    max_swing = (pivot["Ahn"] - pivot["SL"]).abs().max()
    return (f"**Result: max |Ahn − SL| swing in VM/TF median noise ratio = {max_swing:.4f}** "
            f"(static: {pivot.loc['static', 'SL']:.4f} → {pivot.loc['static', 'Ahn']:.4f}; "
            f"temporal: {pivot.loc['temporal', 'SL']:.4f} → {pivot.loc['temporal', 'Ahn']:.4f}).")


def write_report(coverage_lines: list[str], instance_summary: pd.DataFrame, class_summary: pd.DataFrame,
                 concentration: pd.DataFrame, noise_ratio: pd.DataFrame, top12_ratio: pd.DataFrame,
                 flags: list[str], out_path: Path) -> None:
    lines = [
        "# Taxonomy bucket comparison — SL vs. Ahn", "",
        "## Correction (13/08)", "",
        "An earlier version of this report declared the instance-weighted metric "
        "blocked, having missed `ssv2_{vm,tf}_full_detail.parquet` (full "
        "per-(feature,clip) bucket data) during Step 0 and only finding "
        "`class_feature_breakdown.py`'s 4+4-class spot-check output. Both metrics "
        "are now reported. Verified against a known figure: TF's top-12 sign-flip "
        "features here carry 82.546% of TF's sign-flip instances, matching the "
        "brief's cited \"82.5% at L7\" exactly.", "",
        "## Step 0/1 — data coverage", "",
    ] + coverage_lines + [
        "## Step 2A — instance-weighted bucket share (unit = feature-instance)", "",
        df_to_md_table(instance_summary.round(4)), "",
        "## Step 2A — top-12 sign-flip feature concentration", "",
        df_to_md_table(concentration.round(4)), "",
        "## Step 2B — class-weighted bucket proportions (median primary, mean alongside)", "",
        df_to_md_table(class_summary.round(4)), "",
        "## Step 3 — VM/TF noise-fraction ratio (pre-registered falsifier, class-weighted)", "",
        df_to_md_table(noise_ratio.round(4)), "",
        "**Pre-registered expectation (Tom, 12/08, logged before results):** shape "
        "unchanged between taxonomies — the SL/Ahn difference should be small. "
        "**Failure condition:** the VM/TF ratio itself changing, not per-bucket "
        "proportions moving.", "",
        _verdict_line(noise_ratio), "",
        "## Step 3 — VM/TF top-12 sign-flip coverage ratio (instance-weighted)", "",
        df_to_md_table(top12_ratio.round(4)), "",
        _top12_verdict_line(top12_ratio), "",
        "## Instance-weighted vs. class-weighted disagreement", "",
    ] + flags
    out_path.write_text("\n".join(lines))


def vm_tf_noise_ratio(summary: pd.DataFrame) -> pd.DataFrame:
    """Step 3 falsifier — VM/TF ratio of noise fraction, both taxonomies,
    both strata. This is class-weighted only (see module docstring — the
    top-12 sign-flip-coverage ratio needs instance-weighted data, blocked)."""
    noise = summary[summary["bucket"] == "noise"]
    rows = []
    for taxonomy, stratum in noise[["taxonomy", "stratum"]].drop_duplicates().itertuples(index=False):
        sub = noise[(noise["taxonomy"] == taxonomy) & (noise["stratum"] == stratum)].set_index("backbone")
        if not {"vm", "tf"}.issubset(sub.index):
            continue
        rows.append({
            "taxonomy": taxonomy, "stratum": stratum,
            "vm_median_frac_noise": sub.loc["vm", "median_frac"], "tf_median_frac_noise": sub.loc["tf", "median_frac"],
            "vm_tf_ratio_median": sub.loc["vm", "median_frac"] / sub.loc["tf", "median_frac"],
            "vm_tf_ratio_mean": sub.loc["vm", "mean_frac"] / sub.loc["tf", "mean_frac"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    vm = attach_labels(load_backbone("vm", cfg["vm_csv"]), cfg)
    tf = attach_labels(load_backbone("tf", cfg["tf_csv"]), cfg)
    coverage_lines = coverage_report(vm, "vm") + coverage_report(tf, "tf")

    per_class = pd.concat([per_class_fracs(vm), per_class_fracs(tf)], ignore_index=True)
    class_summary = class_weighted_summary(per_class)
    noise_ratio = vm_tf_noise_ratio(class_summary)

    detail = pd.concat([
        load_detail("vm", cfg["vm_detail"], cfg), load_detail("tf", cfg["tf_detail"], cfg),
    ], ignore_index=True)
    instance_summary = instance_weighted_summary(detail)
    concentration = top12_concentration(detail, cfg)
    top12_ratio = vm_tf_top12_ratio(concentration)
    flags = disagreement_flags(concentration)

    out_dir: Path = cfg["out_dir"]
    class_summary.to_csv(out_dir / "bucket_by_taxonomy_class_weighted.csv", index=False)
    instance_summary.to_csv(out_dir / "bucket_by_taxonomy_instance_weighted.csv", index=False)
    write_report(coverage_lines, instance_summary, class_summary, concentration,
                noise_ratio, top12_ratio, flags, out_dir / "taxonomy_bucket_comparison_120826.md")

    print(f"  → {out_dir / 'bucket_by_taxonomy_instance_weighted.csv'}  ({len(instance_summary)} rows)")
    print(f"  → {out_dir / 'bucket_by_taxonomy_class_weighted.csv'}  ({len(class_summary)} rows)")
    print(f"  → {out_dir / 'taxonomy_bucket_comparison_120826.md'}")


if __name__ == "__main__":
    main()
