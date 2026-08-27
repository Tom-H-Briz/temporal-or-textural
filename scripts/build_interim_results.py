"""
Assembles outputs/interim_results/ from verified artifacts only (CC Brief 25/08/26).

Read-only: opens existing parquet/csv/json outputs and counts/filters them.
Never re-runs extraction, ablation, or training. Rerun any time to rebuild the
folder from current disk state.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "interim_results"

# VideoMAE backbone bias fix landed 30/07/2026 (project memory: bias_fix_was_the_model).
# Source files dated before this carry stale/broken VM activations -> PRE-FIX status.
BIAS_FIX_DATE = date(2026, 7, 30)

CENSUS_COLUMNS = [
    "pool", "n", "unit", "weighting", "condition", "config",
    "source_file", "source_mtime", "status",
]

# Per-backbone convention (workbook_entry_240826.md §2, not in this repo -
# brief's own quoted figures used as the record): each backbone drops the SL-35
# classes it fails at <35% top-1 under its own condition-R accuracy file.
VM33_EXCLUDE = {38, 97}       # SL-35 classes VM itself fails at <35% R accuracy
TF32_EXCLUDE = {38, 97, 160}  # SL-35 classes TF itself fails at <35% R accuracy


def mtime_date(path: Path) -> str:
    """ISO date a file was last modified; '' if it doesn't exist."""
    return date.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else ""


def status_for(path: Path, fix_relevant: bool = True, force: str | None = None) -> str:
    """OK / MISSING / PRE-FIX, from disk state and the bias-fix cutoff.

    fix_relevant=False skips the date check entirely: the 30/07 fix changed
    only VideoMAE's model weights (project memory: bias_fix_was_the_model), so
    it cannot have staled (a) TF model-output artifacts - TF was never broken
    by it - or (b) files with no model-computed value at all (clip manifests,
    raw annotation CSVs). Applying the date cutoff to those produces false
    PRE-FIX flags, not real ones.
    """
    if force:
        return force
    if not path.exists():
        return "MISSING"
    if not fix_relevant:
        return "OK"
    mtime = date.fromtimestamp(path.stat().st_mtime)
    return "PRE-FIX" if mtime < BIAS_FIX_DATE else "OK"


def census_row(pool, n, unit, weighting, condition, config, source_file: Path,
                fix_relevant=True, force_status=None):
    """One row in the §2 provenance format, for any pool census table."""
    return {
        "pool": pool, "n": n, "unit": unit, "weighting": weighting,
        "condition": condition, "config": config,
        "source_file": str(source_file), "source_mtime": mtime_date(source_file),
        "status": status_for(source_file, fix_relevant=fix_relevant, force=force_status),
    }


def load_sl35_class_ids() -> set[int]:
    """Canonical SL-35 class id set, read live from its source CSV.

    Per T0: prefer SL-35 over VM-33/TF-32/SL-32 wherever a table's methodology
    allows it - it's the matched, verified-valid population for both backbones.
    """
    path = ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv"
    return set(pd.read_csv(path)["class_id"].tolist())


# ============================== T0: pool census ==============================

# NOTE: the SL-32 raw DFA extraction pool (outputs/dfa/dfa_vm_k64_l7_sl32_290626.parquet
# + dfa_tf_k64_l{3,5,7,9}_sl32_220626.parquet) is intentionally excluded from this
# census. Investigated and removed 25/08: SL-32 is not an analysis population
# anywhere in the current pipeline - DFA mass-delta, ablation, and position-lock
# are all SL-35-based for both backbones (see DISCREPANCIES.md finding 2). The
# extraction pool's only consumer anywhere in src/notebooks is a narrow, unused
# validation notebook. Carrying a dead-end pool here just invites someone to
# treat it as relevant to a report table it doesn't feed.


def rows_ablation() -> list[dict]:
    """Ablation pools: VM (SSv2 L5/L7, K400 L5/L7) + TF (TOP12 sign-flip).

    §3.1(a): settles the audit's internal contradiction. VM's ablation
    parquets carry all 35 SL classes (checked directly below), not the
    32-class SL-32 set the audit's headline finding claimed - the class-scope
    asymmetry it reported does not hold at this pipeline stage.
    """
    rows = []
    vm_configs = [
        ("VM/SSv2/L5/x8k64", "R(spliced)", "ablation_results_long_l5_job7ep_k64.parquet"),
        ("VM/SSv2/L7/x8k64", "R(spliced)", "ablation_results_long_l7_job7ep_k64.parquet"),
        ("VM/K400/L5/x8k64", "R(spliced)", "ablation_results_long_kinetics400_l5_job7ep_k64.parquet"),
        ("VM/K400/L7/x8k64", "R(spliced)", "ablation_results_long_kinetics400_l7_job7ep_k64.parquet"),
    ]
    for config, cond, fname in vm_configs:
        path = ROOT / "outputs/analysis/scaffold_ablation" / fname
        df = pd.read_parquet(path)
        n_cls = df["class_id"].nunique()
        pool = "SL-35" if n_cls == 35 else ("SL-64" if n_cls == 64 else f"UNEXPECTED-{n_cls}cls")
        rows.append(census_row(pool, df["clip_id"].nunique(), "clip", "n/a", cond, config, path))

    tf_path = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_results_long.parquet"
    df = pd.read_parquet(tf_path)
    rows.append(census_row(
        "SL-35", df["clip_id"].nunique(), "clip", "n/a", "R",
        "TF/SSv2/L3+L5+L7+L9/x8k64 (TOP12)", tf_path, fix_relevant=False,
    ))
    return rows


def rows_ablation_row_dropped() -> list[dict]:
    """VM-33 / TF-32: each backbone's ablation pool, row-dropped to the
    classes that backbone itself clears at >=35% R accuracy (§3.1(b)).

    K400 (SL-64 basis) is out of scope for VM-33/TF-32 - those pools are
    SL-35 derivatives and don't apply to the 64-class K400 population.
    """
    rows = []
    vm_configs = [
        ("VM/SSv2/L5/x8k64", "ablation_results_long_l5_job7ep_k64.parquet"),
        ("VM/SSv2/L7/x8k64", "ablation_results_long_l7_job7ep_k64.parquet"),
    ]
    for config, fname in vm_configs:
        path = ROOT / "outputs/analysis/scaffold_ablation" / fname
        df = pd.read_parquet(path)
        kept = df[~df["class_id"].isin(VM33_EXCLUDE)]
        rows.append(census_row("VM-33", kept["clip_id"].nunique(), "clip", "n/a",
                                "R(spliced)", config, path))

    tf_path = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_results_long.parquet"
    df = pd.read_parquet(tf_path)
    kept = df[~df["class_id"].isin(TF32_EXCLUDE)]
    rows.append(census_row("TF-32", kept["clip_id"].nunique(), "clip", "n/a",
                            "R", "TF/SSv2/L3+L5+L7+L9/x8k64 (TOP12)", tf_path,
                            fix_relevant=False))
    return rows


# NOTE: no VM-33/TF-32 row-drop of the DFA extraction pool here - removed
# alongside rows_dfa_extraction() above, same reason (dead-end SL-32 pool).

def rows_condition_accuracy_and_manifests() -> list[dict]:
    """Condition-accuracy pool, K400 SL manifest, K400 val set (audit rows,
    re-opened live rather than copied as literals).

    None of these carry a model-computed value - clip-to-class manifests and
    raw annotation labels only - so the VM bias fix cannot have staled them
    (fix_relevant=False) regardless of file date.
    """
    rows = []
    p = ROOT / "outputs/Laura_SL/manifest_SL_subset.json"
    with open(p) as f:
        d = json.load(f)
    n = sum(len(v) for v in d.values())
    rows.append(census_row("SL-35", n, "clip", "n/a", "n/a", "SSv2 (backbone-agnostic)", p,
                            fix_relevant=False))

    p = ROOT / "outputs/Laura_SL/k400_manifest_SL_subset.json"
    with open(p) as f:
        d = json.load(f)
    n = sum(len(v) for v in d.values()) if isinstance(d, dict) else len(d)
    rows.append(census_row("SL-64", n, "clip", "n/a", "n/a", "K400 (backbone-agnostic)", p,
                            fix_relevant=False))

    p = ROOT / "data/kinetics400/annotations/val.csv"
    df = pd.read_csv(p)
    rows.append(census_row("K400 val", len(df), "clip", "n/a", "n/a", "K400 (raw)", p,
                            fix_relevant=False))
    rows.append(census_row(
        "K400 val (staged)", "19881 (see notes: corroborated from slurm comments, not re-counted - scratch filesystem inaccessible this session)",
        "clip", "n/a", "n/a", "K400 (raw, Isambard scratch)", p, force_status="UNVERIFIED",
    ))
    return rows


def rows_shuffle_composition() -> list[dict]:
    """VM/TF shuffle-composition pools - not re-run per brief §3.1(b)."""
    rows = []
    for backbone, fname, n_classes in [
        ("VM", "ssv2_vm_shuffle_composition.csv", 33),
        ("TF", "ssv2_tf_shuffle_composition.csv", 31),
    ]:
        p = ROOT / "outputs/analysis/shuffle_reduction_composition" / fname
        df = pd.read_csv(p)  # per-class aggregate, not per-clip: sum n_clips for the real clip count
        n = int(df["n_clips"].sum())
        pool = f"SL-{n_classes}"
        note = " (1 class short of TF-32, per 24/08 decision - not re-run)" if backbone == "TF" else ""
        rows.append(census_row(pool + note, n, "clip", "n/a", "R", f"{backbone}/SSv2", p,
                                fix_relevant=(backbone == "VM")))
    return rows


def rows_spliced_accuracy() -> list[dict]:
    """Spliced-accuracy pool - the audit's correction of the 3,098 batch-count error."""
    rows = []
    p = ROOT / "outputs/analysis/spliced_accuracy_sweep/spliced_accuracy_sweep_per_clip.parquet"
    df = pd.read_parquet(p)
    for ds in ("ssv2", "kinetics400"):
        sub = df[df["dataset"] == ds]
        n = sub["clip_id"].nunique()  # multiple configs/layers per clip; count unique clips
        rows.append(census_row(
            "held-out SAE split (not SL-gated)", n, "clip", "n/a", "n/a", ds, p,
        ))
    p = ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv"
    df = pd.read_csv(p)
    rows.append(census_row(
        "ungated SSv2 val", int(df["total"].sum()), "clip", "n/a", "R",
        "VM/SSv2 (R source, refutes the brief's 3,098 batch-count figure)", p,
    ))
    return rows


def build_t0_pool_census() -> pd.DataFrame:
    rows = (
        rows_ablation()
        + rows_ablation_row_dropped()
        + rows_condition_accuracy_and_manifests()
        + rows_shuffle_composition()
        + rows_spliced_accuracy()
    )
    return pd.DataFrame(rows, columns=CENSUS_COLUMNS)


# ================================ shared infra =================================

# Worst-first: a table carrying even one MISSING row is reported MISSING overall.
STATUS_SEVERITY = {"MISSING": 3, "UNVERIFIED": 2, "PRE-FIX": 1, "OK": 0}


def write_manifest(tables: dict[str, tuple[str, pd.DataFrame]]):
    """MANIFEST.csv: one row per output file, its table ID, and worst-case status."""
    rows = []
    for table_id, (filename, df) in tables.items():
        worst = max(df["status"], key=lambda s: STATUS_SEVERITY.get(s, 0))
        rows.append({"table_id": table_id, "output_file": filename,
                      "n_rows": len(df), "status": worst})
    manifest = pd.DataFrame(rows)
    counts = manifest["status"].value_counts()
    summary = ", ".join(f"{counts.get(s, 0)} {s}" for s in ("OK", "MISSING", "UNVERIFIED", "PRE-FIX"))
    with open(OUT_DIR / "MANIFEST.csv", "w") as f:
        f.write(f"# {len(manifest)} tables: {summary}\n")
        manifest.to_csv(f, index=False)
    print(f"MANIFEST.csv: {summary}")


# =========================== TASK 2: P1 condition accuracy =====================

P1_COLUMNS = [
    "backbone", "condition", "weighting", "accuracy", "correct", "total", "n_classes",
    "pool", "n", "unit", "config", "source_file", "source_mtime", "status",
]

# (backbone, condition, filename, fix_relevant) - VM: post-fix-checked dates matter.
# TF: never affected by the VM bias fix (see DISCREPANCIES.md finding 3).
P1_SOURCES = [
    ("VM", "R",  "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv", True),
    ("VM", "C",  "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_C.csv", True),
    ("VM", "C1", "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_C1.csv", True),
    ("VM", "A",  "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_A.csv", True),
    ("TF", "R",  "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv", False),
    ("TF", "C",  "outputs/stage1_class_selection_TF/per_class_accuracy_TF_C.csv", False),
    ("TF", "A",  "outputs/stage1_class_selection_TF/per_class_accuracy_TF_A.csv", False),
]


def build_p1_ssv2_condition_accuracy() -> pd.DataFrame:
    """SSv2 condition accuracy, VM x {R,C,C1,A} and TF x {R,C,A} (TF has no C1 -
    it's VM's own feature-level bucket condition, per brief note). Ungated SL-35,
    both clip-weighted and unweighted (mean-of-classes), per brief §4 P1.
    """
    sl35 = load_sl35_class_ids()
    rows = []
    for backbone, cond, relpath, fix_relevant in P1_SOURCES:
        path = ROOT / relpath
        df = pd.read_csv(path)
        sub = df[df["class_id"].isin(sl35)]
        correct, total = int(sub["correct"].sum()), int(sub["total"].sum())
        for weighting, accuracy in [
            ("clip-weighted", correct / total),
            ("class-weighted", sub["accuracy"].mean()),
        ]:
            rows.append({
                "backbone": backbone, "condition": cond, "weighting": weighting,
                "accuracy": round(accuracy, 4), "correct": correct, "total": total,
                "n_classes": len(sub), "pool": "SL-35", "n": total, "unit": "clip",
                "config": f"{backbone}/SSv2", "source_file": str(path),
                "source_mtime": mtime_date(path),
                "status": status_for(path, fix_relevant=fix_relevant),
            })
    return pd.DataFrame(rows, columns=P1_COLUMNS)


# ============================ TASK 2: P2 K400 condition accuracy ================

P2_COLUMNS = P1_COLUMNS  # same shape; population column values differ (K400, not SSv2)

# VM only - no TF K400 condition-accuracy file exists anywhere in the repo.
# B and C are listed MISSING deliberately: no file for either exists under
# outputs/stage1_class_selection_VM_kinetics/ or anywhere else checked.
P2_SOURCES = [
    ("R",  "outputs/stage1_class_selection_VM_kinetics/per_class_accuracy_VM_kinetics_R.csv"),
    ("C1", "outputs/stage1_class_selection_VM_kinetics/per_class_accuracy_VM_kinetics_C1.csv"),
    ("A",  "outputs/stage1_class_selection_VM_kinetics/per_class_accuracy_VM_kinetics_A.csv"),
    ("B",  None),
    ("C",  None),
]


def load_sl64_class_ids() -> set[int]:
    """Canonical K400 SL-64 class id set (matched_model_class_id, dropna)."""
    path = ROOT / "outputs/Laura_SL/k400_sl_class_mapping.csv"
    df = pd.read_csv(path).dropna(subset=["matched_model_class_id"])
    return set(df["matched_model_class_id"].astype(int).tolist())


def build_p2_k400_condition_accuracy() -> pd.DataFrame:
    """K400 condition accuracy, VM x {R,C1,A} (B,C reported MISSING - no artifact
    exists). Both ungated (full 400-class val, matches workbook_entry_190826.md's
    cited 76.83/73.73/54.98 exactly) and SL-64-gated cuts. Per brief §4 P2: report
    what exists, log the B/C-vs-A contradiction, don't reconcile by choosing.
    """
    sl64 = load_sl64_class_ids()
    rows = []
    for cond, relpath in P2_SOURCES:
        if relpath is None:
            rows.append({
                "backbone": "VM", "condition": cond, "weighting": "n/a",
                "accuracy": None, "correct": None, "total": None, "n_classes": None,
                "pool": "K400 val", "n": None, "unit": "clip", "config": "VM/K400",
                "source_file": None, "source_mtime": None, "status": "MISSING",
            })
            continue
        path = ROOT / relpath
        df = pd.read_csv(path)
        for pool_label, sub in [("K400 val", df), ("SL-64", df[df["class_id"].isin(sl64)])]:
            correct, total = int(sub["correct"].sum()), int(sub["total"].sum())
            for weighting, accuracy in [
                ("clip-weighted", correct / total),
                ("class-weighted", sub["accuracy"].mean()),
            ]:
                rows.append({
                    "backbone": "VM", "condition": cond, "weighting": weighting,
                    "accuracy": round(accuracy, 4), "correct": correct, "total": total,
                    "n_classes": len(sub), "pool": pool_label, "n": total, "unit": "clip",
                    "config": "VM/K400", "source_file": str(path),
                    "source_mtime": mtime_date(path), "status": status_for(path),
                })
    return pd.DataFrame(rows, columns=P2_COLUMNS)


# =========================== TASK 2: P3 spliced accuracy ========================

P3_COLUMNS = [
    "dataset", "config", "layer", "sae_k", "n", "baseline_correct", "baseline_accuracy",
    "spliced_correct", "spliced_accuracy", "delta_accuracy",
    "pool", "weighting", "condition", "unit", "source_file", "source_mtime", "status",
]


def build_p3_spliced_accuracy() -> pd.DataFrame:
    """Spliced accuracy headline, per backbone/layer/config (brief §4 P3). VM
    only - held-out SAE-training split, not SL-gated (see T0/DISCREPANCIES.md
    finding on the brief's refuted 3,098 batch-count figure).
    """
    path = ROOT / "outputs/analysis/spliced_accuracy_sweep/spliced_accuracy_sweep_per_clip.parquet"
    df = pd.read_parquet(path)
    g = df.groupby(["dataset", "config", "layer", "sae_k"]).agg(
        n=("clip_id", "nunique"),
        baseline_correct=("baseline_correct", "sum"),
        spliced_correct=("spliced_correct", "sum"),
    ).reset_index()
    g["baseline_accuracy"] = (g["baseline_correct"] / g["n"]).round(4)
    g["spliced_accuracy"] = (g["spliced_correct"] / g["n"]).round(4)
    g["delta_accuracy"] = (g["spliced_accuracy"] - g["baseline_accuracy"]).round(4)
    g["pool"] = "held-out SAE split (not SL-gated)"
    g["weighting"] = "clip-weighted"
    g["condition"] = "R (baseline) vs R+SAE-splice"
    g["unit"] = "clip"
    g["source_file"] = str(path)
    g["source_mtime"] = mtime_date(path)
    g["status"] = status_for(path)
    return g[P3_COLUMNS]


# ============================ TASK 2: A1 scaffold member counts =================

A1_COLUMNS = [
    "config", "backbone", "layer", "sae_k", "dataset", "n_members",
    "n_ablation_targets", "member_combined_mass_R", "ceiling_combined_mass_R",
    "pct_of_ceiling", "member_combined_mass_R_canonical",
    "ceiling_combined_mass_R_canonical", "pct_of_ceiling_canonical",
    "pool", "unit", "weighting", "condition", "config_full",
    "source_file", "source_mtime", "status",
]

# Ablation target counts, verified directly from each ablation_results_long
# parquet's unique `ablation_target` values (single_* entries). Only L5/L7 have
# ablation runs on disk; K400 L5/L7 match their gate n_members exactly (10, 6),
# SSv2 L5 matches too (7) - only SSv2 L7 has the near-miss discrepancy (gate 3,
# ablation 4, feature 3347 included as a narrow C1 near-miss). Not populated
# where no ablation parquet exists.
A1_ABLATION_TARGETS = {
    "L5_x8k64_VM": 7, "L7_x8k64_VM": 4,
    "L5_x8k64_VM_K400": 10, "L7_x8k64_VM_K400": 6,
}

# Per user (25/08): the actually-ablated feature set is the canonical scaffold,
# not the raw gate output - for L7_x8k64_VM that's gate (3) + the one near-miss
# (3347) that ablation also used, matching n_ablation_targets above. Source of
# the feature list: ablation_results_long's own `single_*` targets, read live
# below, not hardcoded, so it can't drift if a config's ablation set changes.
A1_ABLATION_RESULTS_PATHS = {
    "L5_x8k64_VM": "outputs/analysis/scaffold_ablation/ablation_results_long_l5_job7ep_k64.parquet",
    "L7_x8k64_VM": "outputs/analysis/scaffold_ablation/ablation_results_long_l7_job7ep_k64.parquet",
    "L5_x8k64_VM_K400": "outputs/analysis/scaffold_ablation/ablation_results_long_kinetics400_l5_job7ep_k64.parquet",
    "L7_x8k64_VM_K400": "outputs/analysis/scaffold_ablation/ablation_results_long_kinetics400_l7_job7ep_k64.parquet",
}


def _canonical_ablated_features(config_name: str) -> list[int] | None:
    """Feature ids actually ablated for a config, from its ablation_results_long
    parquet's single_* targets - the canonical scaffold set (may include a
    near-miss the raw gate excluded, e.g. L7_x8k64_VM's feature 3347)."""
    relpath = A1_ABLATION_RESULTS_PATHS.get(config_name)
    if relpath is None:
        return None
    df = pd.read_parquet(ROOT / relpath)
    singles = [t for t in df["ablation_target"].unique() if t.startswith("single_")]
    return sorted(int(t.removeprefix("single_")) for t in singles)


def build_a1_scaffold_member_counts() -> pd.DataFrame:
    """Member counts per layer/dataset/SAE config (brief §4 A1). Reads the
    CONFIGS list directly from scaffold_selection_consolidated.py so this can't
    drift out of sync with what the pipeline actually defines. L9 = 0 rows kept
    explicit (real, computed zeros). TF configs reported MISSING, not silently
    dropped - scaffold_selection_consolidated.py defines 3 TF configs but skips
    them at runtime because their position_lock_scores input was never
    generated (T0/DISCREPANCIES.md finding 3: TF position-lock never rerun).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scaffold_selection_consolidated", ROOT / "src/stage3_analysis/scaffold_selection_consolidated.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ceiling_path = ROOT / "outputs/analysis/scaffold_selection/scaffold_selection_ceiling_summary.csv"
    ceiling = pd.read_csv(ceiling_path).set_index("config")

    expansion_for_k = {64: 8, 128: 16}  # project convention: k64 pairs with x8, k128 with x16
    rows = []
    for c in mod.CONFIGS:
        name = c["name"]
        backbone = "VM" if c["model"] == "videomae" else "TF"
        if name in ceiling.index:
            r = ceiling.loc[name]
            n_members = int(r["n_members"])
            status = "OK"
            member_mass, ceiling_mass, pct = r["member_combined_mass_R"], r["ceiling_combined_mass_R"], r["pct_of_ceiling"]
        else:
            n_members, member_mass, ceiling_mass, pct = None, None, None, None
            status = "MISSING"

        canonical = _canonical_ablated_features(name)
        member_mass_c, ceiling_mass_c, pct_c = member_mass, ceiling_mass, pct
        if canonical and status == "OK":
            try:
                mass_delta_path = mod._resolve_mass_delta_parquet(c)
                mat = np.stack(pd.read_parquet(mass_delta_path)["signed_vec_R"].to_numpy()).astype(np.float32)
                ranked = np.argsort(mod._raw_feature_masses(mat))[::-1]
                ceiling_features = ranked[:len(canonical)].tolist()
                member_mass_c = mod._combined_mass_pct_r(mat, canonical)
                ceiling_mass_c = mod._combined_mass_pct_r(mat, ceiling_features)
                pct_c = member_mass_c / ceiling_mass_c * 100
            except FileNotFoundError:
                pass

        rows.append({
            "config": name, "backbone": backbone, "layer": c["layer"], "sae_k": c["sae_k"],
            "dataset": c["dataset"], "n_members": n_members,
            "n_ablation_targets": A1_ABLATION_TARGETS.get(name),
            "member_combined_mass_R": member_mass, "ceiling_combined_mass_R": ceiling_mass,
            "pct_of_ceiling": pct,
            "member_combined_mass_R_canonical": member_mass_c,
            "ceiling_combined_mass_R_canonical": ceiling_mass_c,
            "pct_of_ceiling_canonical": pct_c,
            "pool": "SL-35" if c["dataset"] == "ssv2" else "SL-64",
            "unit": "feature", "weighting": "n/a", "condition": "R",
            "config_full": f"{backbone}/{c['dataset']}/L{c['layer']}/x{expansion_for_k[c['sae_k']]}k{c['sae_k']}",
            "source_file": str(ceiling_path) if status == "OK" else None,
            "source_mtime": mtime_date(ceiling_path) if status == "OK" else None,
            "status": status,
        })
    return pd.DataFrame(rows, columns=A1_COLUMNS)


# ======================= TASK 2: A2 mass concentration bootstrap ================

A2_COLUMNS = [
    "config", "backbone", "n_members", "dict_size", "member_mass_pct",
    "boot_mean_pct", "boot_max_pct", "best_draw_ratio", "n_draws",
    "exceedance_count", "p_bound", "pool", "unit", "weighting", "condition",
    "source_file", "source_mtime", "status",
]

N_BOOTSTRAP_DRAWS = 2000  # CFG["n_draws"] in mass_concentration_bootstrap.py


def _bootstrap_canonical(mod, config_name: str, canonical: list[int]) -> dict:
    """Reproduces mass_concentration_bootstrap.py's bootstrap_config() exactly
    (same RNG scheme: seed=0 + crc32(config_name) salt, same n_draws=2000),
    just with the canonical (actually-ablated) feature list instead of the
    raw gate's member list. Not a new method - the same documented procedure,
    applied to the corrected feature set, same as A1/A3's ceiling recompute.
    """
    import zlib
    cfg = next(c for c in mod.CONFIGS if c["name"] == config_name)
    mat = np.stack(pd.read_parquet(mod._resolve_mass_delta_parquet(cfg))["signed_vec_R"].to_numpy()).astype(np.float32)
    dict_size = mat.shape[1]
    member_mass = mod._combined_mass_pct_r(mat, canonical)
    rng = np.random.default_rng([0, zlib.crc32(config_name.encode())])
    draws = np.array([
        mod._combined_mass_pct_r(mat, rng.choice(dict_size, size=len(canonical), replace=False).tolist())
        for _ in range(N_BOOTSTRAP_DRAWS)
    ])
    return {"n_members": len(canonical), "member_mass_pct": member_mass * 100,
            "boot_mean_pct": draws.mean() * 100, "boot_max_pct": draws.max() * 100,
            "exceedance_count": int((draws >= member_mass).sum())}


def build_a2_mass_concentration_bootstrap() -> pd.DataFrame:
    """Best-draw ratio + exceedance count only (brief §4 A2) - no z-scores,
    even though the source file has one. x16k128 mass rows are OK, not
    UNVERIFIED: open item 2 is resolved (see DISCREPANCIES.md) - the 13/08
    fix script (commit c12fceb) reran mass-delta for k128 post-fix, and
    scaffold_selection_consolidated.py's resolver prefers that fresh file.
    p_bound is "<1/{n_draws}" rather than a literal 0 - 0/2000 exceedances
    bounds the true p-value above, it doesn't measure it as exactly zero.

    Canonical columns added per user (25/08), same reasoning as A1/A3: the
    raw gate output isn't always the actually-ablated set (L7_x8k64_VM: gate
    3, ablation 4, including near-miss feature 3347) - recomputed with the
    same documented bootstrap method, canonical feature list, for any config
    where the two differ.
    """
    path = ROOT / "outputs/analysis/mass_concentration_bootstrap/mass_concentration_bootstrap.csv"
    df = pd.read_csv(path)
    df["exceedance_count"] = (df["pct_random_draws_beating_scaffold"] / 100 * N_BOOTSTRAP_DRAWS).round().astype(int)
    df["backbone"] = "VM"  # bootstrap script only covers VM configs (mass_concentration_bootstrap.py's CONFIGS import)
    df["best_draw_ratio"] = (df["member_mass_pct"] / df["boot_max_pct"]).round(3)
    df["n_draws"] = N_BOOTSTRAP_DRAWS
    df["p_bound"] = df["exceedance_count"].apply(lambda c: "<1/2000" if c == 0 else f"{c}/2000")
    df["pool"] = df["config"].apply(lambda c: "SL-64" if c.endswith("K400") else "SL-35")
    df["unit"] = "(feature,clip)"
    df["weighting"] = "n/a"
    df["condition"] = "R"
    df["source_file"] = str(path)
    df["source_mtime"] = mtime_date(path)
    df["status"] = status_for(path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scaffold_selection_consolidated", ROOT / "src/stage3_analysis/scaffold_selection_consolidated.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    canon_cols = ["n_members_canonical", "member_mass_pct_canonical", "boot_mean_pct_canonical",
                  "boot_max_pct_canonical", "best_draw_ratio_canonical", "exceedance_count_canonical",
                  "p_bound_canonical"]
    for col in canon_cols:
        df[col] = df[col.removesuffix("_canonical")] if col.removesuffix("_canonical") in df.columns else None
    for idx, row in df.iterrows():
        canonical = _canonical_ablated_features(row["config"])
        if canonical and len(canonical) != row["n_members"]:
            r = _bootstrap_canonical(mod, row["config"], canonical)
            df.loc[idx, "n_members_canonical"] = r["n_members"]
            df.loc[idx, "member_mass_pct_canonical"] = r["member_mass_pct"]
            df.loc[idx, "boot_mean_pct_canonical"] = r["boot_mean_pct"]
            df.loc[idx, "boot_max_pct_canonical"] = r["boot_max_pct"]
            df.loc[idx, "best_draw_ratio_canonical"] = round(r["member_mass_pct"] / r["boot_max_pct"], 3)
            df.loc[idx, "exceedance_count_canonical"] = r["exceedance_count"]
            df.loc[idx, "p_bound_canonical"] = "<1/2000" if r["exceedance_count"] == 0 else f"{r['exceedance_count']}/2000"
    return df[A2_COLUMNS + canon_cols]


# ============================ TASK 2: A3 ceiling occupancy ======================

A3_COLUMNS = [
    "config", "backbone", "layer", "sae_k", "dataset", "dict_size", "n_members",
    "metric", "value", "cross_config_comparable", "pool", "unit", "weighting",
    "condition", "source_file", "source_mtime", "status",
]


def build_a3_ceiling_occupancy() -> pd.DataFrame:
    """Ceiling occupancy (cross-config comparable) + enrichment ratio
    (within-config only) as separate, explicitly-flagged rows per brief §4 A3 -
    the caveat lives in a `cross_config_comparable` column on every row rather
    than being left to a footnote, so it can't be dropped by a downstream
    reshape. Enrichment ratio = member_mass_pct / (100 * n_members/dict_size):
    how many times the scaffold's actual mass share exceeds the proportional
    share it would get if mass were spread uniformly across the dictionary.
    """
    a1 = build_a1_scaffold_member_counts()
    boot_path = ROOT / "outputs/analysis/mass_concentration_bootstrap/mass_concentration_bootstrap.csv"
    boot = pd.read_csv(boot_path).set_index("config")[["dict_size", "member_mass_pct"]]

    rows = []
    for _, r in a1.iterrows():
        common = {"config": r["config"], "backbone": r["backbone"], "layer": r["layer"],
                   "sae_k": r["sae_k"], "dataset": r["dataset"], "pool": r["pool"],
                   "unit": "n/a", "weighting": "n/a", "condition": "R"}
        if r["status"] == "MISSING":
            for metric in ("pct_of_ceiling", "enrichment_ratio"):
                rows.append({**common, "dict_size": None, "n_members": None, "metric": metric,
                             "value": None, "cross_config_comparable": metric == "pct_of_ceiling",
                             "source_file": None, "source_mtime": None, "status": "MISSING"})
            continue
        dict_size = int(boot.loc[r["config"], "dict_size"]) if r["config"] in boot.index else None
        mass_pct = boot.loc[r["config"], "member_mass_pct"] if r["config"] in boot.index else None
        enrichment = (mass_pct / (100 * r["n_members"] / dict_size)) if dict_size and r["n_members"] else None
        rows.append({**common, "dict_size": dict_size, "n_members": r["n_members"],
                     "metric": "pct_of_ceiling", "value": round(r["pct_of_ceiling"], 3),
                     "cross_config_comparable": True, "source_file": r["source_file"],
                     "source_mtime": r["source_mtime"], "status": r["status"]})
        rows.append({**common, "dict_size": dict_size, "n_members": r["n_ablation_targets"] or r["n_members"],
                     "metric": "pct_of_ceiling_canonical (actually-ablated feature set, not just gate)",
                     "value": round(r["pct_of_ceiling_canonical"], 3) if pd.notna(r["pct_of_ceiling_canonical"]) else None,
                     "cross_config_comparable": True, "source_file": r["source_file"],
                     "source_mtime": r["source_mtime"], "status": r["status"]})
        rows.append({**common, "dict_size": dict_size, "n_members": r["n_members"],
                     "metric": "enrichment_ratio", "value": round(enrichment, 3) if enrichment else None,
                     "cross_config_comparable": False,
                     "source_file": str(boot_path) if enrichment else None,
                     "source_mtime": mtime_date(boot_path) if enrichment else None,
                     "status": "OK" if enrichment else "MISSING"})
    return pd.DataFrame(rows, columns=A3_COLUMNS)


# ========================= TASK 2: A4 scaffold ablation additivity ==============

A4_COLUMNS = [
    "config", "backbone", "condition", "group_target",
    "singleton_sum_mean_logit_delta", "group_delta_mean_logit_delta",
    "additivity_gap_mean_logit_delta", "ratio_group_over_singleton",
    "pool", "unit", "weighting", "source_file", "source_mtime", "status",
]

# VM: current job7ep configs only - the unsuffixed outputs/analysis/scaffold_ablation/
# ablation_additivity.csv (02/07, pre-fix, no job7ep tag) is superseded by these
# per-config files for the same layers and excluded, same reasoning as T0
# finding 2 (dead pre-fix duplicate with a live successor, not carried forward).
A4_VM_SOURCES = [
    ("VM/SSv2/L5/x8k64", "SL-35", "outputs/analysis/scaffold_ablation/ablation_additivity_l5_job7ep_k64.csv"),
    ("VM/SSv2/L7/x8k64", "SL-35", "outputs/analysis/scaffold_ablation/ablation_additivity_l7_job7ep_k64.csv"),
    ("VM/K400/L5/x8k64", "SL-64", "outputs/analysis/scaffold_ablation/ablation_additivity_kinetics400_l5_job7ep_k64.csv"),
    ("VM/K400/L7/x8k64", "SL-64", "outputs/analysis/scaffold_ablation/ablation_additivity_kinetics400_l7_job7ep_k64.csv"),
]


def build_a4_ablation_additivity() -> pd.DataFrame:
    """Singleton sum / group delta / additivity gap / ratio, R and C1 (VM) or
    R and C (TF, no C1) - brief §4 A4. Metric is mean logit delta, stated in
    the column names themselves, not accuracy.
    """
    rows = []
    for config, pool, relpath in A4_VM_SOURCES:
        path = ROOT / relpath
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "config": config, "backbone": "VM", "condition": r["perturbation_condition"],
                "group_target": r["group_target"],
                "singleton_sum_mean_logit_delta": r["singleton_sum"],
                "group_delta_mean_logit_delta": r["group_delta"],
                "additivity_gap_mean_logit_delta": r["additivity_gap"],
                "ratio_group_over_singleton": round(r["group_delta"] / r["singleton_sum"], 3) if r["singleton_sum"] else None,
                "pool": pool, "unit": "(feature,clip)", "weighting": "n/a",
                "source_file": str(path), "source_mtime": mtime_date(path), "status": status_for(path),
            })

    tf_path = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_additivity.csv"
    tf = pd.read_csv(tf_path)
    for _, r in tf.iterrows():
        rows.append({
            "config": "TF/SSv2/L3+L5+L7+L9/x8k64 (TOP12)", "backbone": "TF",
            "condition": r["perturbation_condition"], "group_target": "top5",
            "singleton_sum_mean_logit_delta": r["singleton_sum"],
            "group_delta_mean_logit_delta": r["top5_delta"],
            "additivity_gap_mean_logit_delta": r["additivity_gap"],
            "ratio_group_over_singleton": round(r["top5_delta"] / r["singleton_sum"], 3) if r["singleton_sum"] else None,
            "pool": "SL-35", "unit": "(feature,clip)", "weighting": "n/a",
            "source_file": str(tf_path), "source_mtime": mtime_date(tf_path),
            "status": status_for(tf_path, fix_relevant=False),
        })
    return pd.DataFrame(rows, columns=A4_COLUMNS)


# ==================== TASK 2: A5 temporal/static ablation differential ==========

A5_COLUMNS = [
    "config", "backbone", "taxonomy", "condition", "group_target",
    "static_mean_logit_delta", "temporal_mean_logit_delta", "overall_mean_logit_delta",
    "unassigned_mean_logit_delta", "differential_static_minus_temporal",
    "n_clips_static", "n_clips_temporal", "pool", "unit", "weighting",
    "source_file", "source_mtime", "status",
]

# (config, pool, group_target, path) - group_target picks the joint-ablation
# row out of each file's per-target rows (matches A4's group targets).
A5_VM_SL_SOURCES = [
    ("VM/SSv2/L5/x8k64", "SL-35", "all7", "outputs/analysis/scaffold_ablation/ablation_summary_l5_job7ep_k64.csv"),
    ("VM/SSv2/L7/x8k64", "SL-35", "all4", "outputs/analysis/scaffold_ablation/ablation_summary_l7_job7ep_k64.csv"),
    ("VM/K400/L5/x8k64", "SL-64", "all_members", "outputs/analysis/scaffold_ablation/ablation_summary_kinetics400_l5_job7ep_k64.csv"),
    ("VM/K400/L7/x8k64", "SL-64", "all_members", "outputs/analysis/scaffold_ablation/ablation_summary_kinetics400_l7_job7ep_k64.csv"),
]
# Only L5 has an Ahn-32 ablation summary on disk - L7/K400 reported MISSING.
A5_VM_AHN_SOURCES = [
    ("VM/SSv2/L5/x8k64", "SL-35", "all7", "outputs/analysis/scaffold_ablation/ablation_summary_l5_job7ep_k64_ahn32.csv"),
]
A5_VM_AHN_MISSING = ["VM/SSv2/L7/x8k64", "VM/K400/L5/x8k64", "VM/K400/L7/x8k64"]


def _a5_row_from_group(config, backbone, taxonomy, pool, group_target, path):
    df = pd.read_csv(path)
    group = df[df["ablation_target"] == group_target]
    rows = []
    for cond, g in group.groupby("perturbation_condition"):
        by_label = g.set_index("sl_label")
        get = lambda lbl, col: by_label.loc[lbl, col] if lbl in by_label.index else None
        static, temporal = get("static", "mean_delta"), get("temporal", "mean_delta")
        rows.append({
            "config": config, "backbone": backbone, "taxonomy": taxonomy, "condition": cond,
            "group_target": group_target, "static_mean_logit_delta": static,
            "temporal_mean_logit_delta": temporal, "overall_mean_logit_delta": get("overall", "mean_delta"),
            "unassigned_mean_logit_delta": get("unassigned", "mean_delta"),
            "differential_static_minus_temporal": round(static - temporal, 4) if static is not None and temporal is not None else None,
            "n_clips_static": get("static", "n_clips"), "n_clips_temporal": get("temporal", "n_clips"),
            "pool": pool, "unit": "(feature,clip)", "weighting": "n/a",
            "source_file": str(path), "source_mtime": mtime_date(path), "status": status_for(path),
        })
    return rows


def build_a5_temporal_static_ablation() -> pd.DataFrame:
    """Temporal/static differential under ablation, SL and Ahn taxonomies, both
    conditions (brief §4 A5). Uses each source's own group-ablation row (all7/
    all4/all_members/TOP12), matching A4's group targets.
    """
    rows = []
    for config, pool, target, relpath in A5_VM_SL_SOURCES:
        rows += _a5_row_from_group(config, "VM", "SL", pool, target, ROOT / relpath)
    for config, pool, target, relpath in A5_VM_AHN_SOURCES:
        rows += _a5_row_from_group(config, "VM", "Ahn", pool, target, ROOT / relpath)
    for config in A5_VM_AHN_MISSING:
        rows.append({
            "config": config, "backbone": "VM", "taxonomy": "Ahn", "condition": None,
            "group_target": None, "static_mean_logit_delta": None, "temporal_mean_logit_delta": None,
            "overall_mean_logit_delta": None, "unassigned_mean_logit_delta": None,
            "differential_static_minus_temporal": None, "n_clips_static": None, "n_clips_temporal": None,
            "pool": "SL-35" if "SSv2" in config else "SL-64", "unit": "(feature,clip)", "weighting": "n/a",
            "source_file": None, "source_mtime": None, "status": "MISSING",
        })

    tf_path = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_summary.csv"
    tf = pd.read_csv(tf_path)
    tf_group = tf[tf["ablation_target"] == "TOP12"]
    for _, r in tf_group.iterrows():
        rows.append({
            "config": "TF/SSv2/L7/x8k64 (TOP12)", "backbone": "TF", "taxonomy": "SL",
            "condition": r["perturbation_condition"], "group_target": "TOP12",
            "static_mean_logit_delta": r["static_delta"], "temporal_mean_logit_delta": r["temporal_delta"],
            "overall_mean_logit_delta": None, "unassigned_mean_logit_delta": None,
            "differential_static_minus_temporal": round(r["static_delta"] - r["temporal_delta"], 4),
            "n_clips_static": r["n_clips_static"], "n_clips_temporal": r["n_clips_temporal"],
            "pool": "SL-35", "unit": "(feature,clip)", "weighting": "n/a",
            "source_file": str(tf_path), "source_mtime": mtime_date(tf_path),
            "status": status_for(tf_path, fix_relevant=False),
        })
    rows.append({
        "config": "TF/SSv2/L7/x8k64 (TOP12)", "backbone": "TF", "taxonomy": "Ahn", "condition": None,
        "group_target": None, "static_mean_logit_delta": None, "temporal_mean_logit_delta": None,
        "overall_mean_logit_delta": None, "unassigned_mean_logit_delta": None,
        "differential_static_minus_temporal": None, "n_clips_static": None, "n_clips_temporal": None,
        "pool": "SL-35", "unit": "(feature,clip)", "weighting": "n/a",
        "source_file": None, "source_mtime": None, "status": "MISSING",
    })
    return pd.DataFrame(rows, columns=A5_COLUMNS)


# ============================= TASK 2: A6 per-class alignment ===================

A6_COLUMNS = [
    "class_id", "template", "n_clips", "vm_accuracy", "tf_accuracy",
    "delta_unablated_median", "delta_unablated_ci_lo", "delta_unablated_ci_hi",
    "delta_l5_median", "delta_l5_ci_lo", "delta_l5_ci_hi",
    "alignment_median", "alignment_ci_lo", "alignment_ci_hi", "unstable",
    "pool", "unit", "weighting", "condition", "source_file", "source_mtime", "status",
]


def build_a6_per_class_alignment() -> pd.DataFrame:
    """VM, VM-ablated, TF, alignment ratio, bootstrap CI, per class (brief §4
    A6). Uses the clip-paired bootstrap file (35 classes, full SL-35 - better
    than the brief-named 32-class point-estimate file, per 'use 35 matching
    wherever we can'), not the independent-resample variant (paired is the
    correct method - VM and TF are scored on identical clips per class).
    Full table, appendix - anchor rows selected later, not here.
    """
    path = ROOT / "outputs/analysis/scaffold_ablation/vm_tf_accuracy_vs_l5_ablation.csv"
    df = pd.read_csv(path)
    out = df[["class_id", "template", "n_clips", "vm_accuracy", "tf_accuracy",
              "delta_unablated_median", "delta_unablated_ci_lo", "delta_unablated_ci_hi",
              "delta_l5_median", "delta_l5_ci_lo", "delta_l5_ci_hi",
              "alignment_median", "alignment_ci_lo", "alignment_ci_hi", "unstable"]].copy()
    out["pool"] = "SL-35"
    out["unit"] = "class"
    out["weighting"] = "n/a"
    out["condition"] = "R vs L5-ablated"
    out["source_file"] = str(path)
    out["source_mtime"] = mtime_date(path)
    out["status"] = status_for(path)
    return out[A6_COLUMNS]


# ========================= TASK 2: A7 TF near-miss distribution =================

A7_COLUMNS = [
    "config", "backbone", "layer", "dataset", "row_type", "metric", "percentile",
    "feature_idx", "member_status", "share_value", "n_features", "pool",
    "unit", "weighting", "condition", "source_file", "source_mtime", "status",
]

A7_PERCENTILES = [50, 90, 95, 99, 99.9]

VM_SSV2_CONFIGS = [
    ("L3_x8k64_VM", 3, "ssv2"), ("L5_x8k64_VM", 5, "ssv2"), ("L7_x8k64_VM", 7, "ssv2"),
    ("L5_x16k128_VM", 5, "ssv2"), ("L7_x16k128_VM", 7, "ssv2"),
    ("L9_x8k64_VM", 9, "ssv2"), ("L9_x16k128_VM", 9, "ssv2"),
]
VM_K400_CONFIGS = [
    ("L3_x8k64_VM_K400", 3, "kinetics400"), ("L5_x8k64_VM_K400", 5, "kinetics400"),
    ("L7_x8k64_VM_K400", 7, "kinetics400"), ("L9_x8k64_VM_K400", 9, "kinetics400"),
]


def build_a7_near_miss_distribution() -> pd.DataFrame:
    """Whole-dictionary share percentiles + VM floor per config, plus the
    SSv2 L7 near-miss detail (brief §4 A7).

    TF is now populated (corrected 25/08 - an earlier version of this table
    reported TF MISSING). Per DISCREPANCIES.md findings 3/10: the activity-gate
    mask bug only ever affected the DFA-quantity accumulation
    (dfa_per_tubelet_mass.py) - the z-quantity (raw-activation) accumulation in
    z_position_lock_extraction.py already had the mask, confirmed by the merged
    script's own docstring, and the DFA-quantity bug was minor anyway (user,
    25/08). TF's z-quantity files
    (outputs/analysis/z_position_lock/z_position_lock_scores_timesformer_l{5,7,9}.csv)
    are used here. VM is reported on BOTH metrics (share_R_dfa - the metric the
    current scaffold gate actually uses - and share_R_z, matched to what's
    available for TF) so the TF-vs-VM comparison is apples-to-apples on
    share_R_z specifically, not mixed across differently-computed quantities.
    """
    rows = []
    for name, layer, dataset in VM_SSV2_CONFIGS + VM_K400_CONFIGS:
        path = ROOT / "outputs/analysis/scaffold_selection" / f"{name}_all_features.csv"
        df = pd.read_csv(path)
        pool = "SL-35" if dataset == "ssv2" else "SL-64"
        common = {"config": name, "backbone": "VM", "layer": layer, "dataset": dataset,
                  "pool": pool, "unit": "feature", "weighting": "n/a", "condition": "R",
                  "source_file": str(path), "source_mtime": mtime_date(path), "status": status_for(path)}
        for metric, col in [("share_R_dfa", "share_R_dfa"), ("share_R_z", "share_R_z")]:
            for pct in A7_PERCENTILES:
                rows.append({**common, "row_type": "whole_dict_percentile", "metric": metric,
                             "percentile": pct, "feature_idx": None, "member_status": None,
                             "share_value": round(df[col].quantile(pct / 100), 4), "n_features": len(df)})
            members = df[df["status"] == "member"]
            floor = members[col].min() if len(members) else None
            rows.append({**common, "row_type": "vm_floor", "metric": metric, "percentile": None,
                         "feature_idx": None, "member_status": "member",
                         "share_value": round(floor, 4) if floor is not None else None,
                         "n_features": len(members)})

    l7_members_path = ROOT / "outputs/analysis/scaffold_selection/L7_x8k64_VM.csv"
    l7 = pd.read_csv(l7_members_path)
    for _, r in l7.iterrows():
        rows.append({
            "config": "L7_x8k64_VM", "backbone": "VM", "layer": 7, "dataset": "ssv2",
            "row_type": "l7_natural_break_detail", "metric": "share_R_dfa", "percentile": None,
            "feature_idx": r["feature_idx"], "member_status": r["status"],
            "share_value": round(r["dfa_share_R"], 4), "n_features": None,
            "pool": "SL-35", "unit": "feature", "weighting": "n/a", "condition": "R",
            "source_file": str(l7_members_path), "source_mtime": mtime_date(l7_members_path),
            "status": status_for(l7_members_path),
        })

    for layer in (5, 7, 9):
        path = ROOT / "outputs/analysis/z_position_lock" / f"z_position_lock_scores_timesformer_l{layer}.csv"
        df = pd.read_csv(path)
        collapsed = df.groupby("feature_idx")["mean_per_clip_share_R"].min()  # min-across-classes, matches the gate rule
        common = {"config": f"L{layer}_x8k64_TF", "backbone": "TF", "layer": layer, "dataset": "ssv2",
                  "row_type": "whole_dict_percentile", "metric": "share_R_z", "feature_idx": None,
                  "member_status": None, "n_features": len(collapsed), "pool": "SL-32",
                  "unit": "feature", "weighting": "n/a", "condition": "R",
                  "source_file": str(path), "source_mtime": mtime_date(path), "status": status_for(path, fix_relevant=False)}
        for pct in A7_PERCENTILES:
            rows.append({**common, "percentile": pct, "share_value": round(collapsed.quantile(pct / 100), 4)})
        rows.append({**common, "row_type": "tf_whole_dict_max", "percentile": None,
                     "share_value": round(collapsed.max(), 4)})
    return pd.DataFrame(rows, columns=A7_COLUMNS)


# ========================= TASK 2: B1 four-bucket fractions =====================

B1_COLUMNS = [
    "backbone", "taxonomy", "stratum", "n_classes", "n_clips",
    "frac_noise", "frac_sign_flip", "frac_decrease", "frac_increase",
    "pool", "unit", "weighting", "condition", "source_file", "source_mtime", "status",
]

BUCKET_COLS = ["frac_noise", "frac_sign_flip", "frac_decrease", "frac_increase"]


def _load_taxonomy_labels() -> pd.DataFrame:
    """SL label (35 classes) + Ahn label (32 classes, unassigned for the
    3 SL-32-excluded classes when present) per class_id, from their
    respective canonical sources."""
    sl = pd.read_csv(ROOT / "outputs/Laura_SL/accuracy_SL_subset.csv")[["class_id", "category"]]
    sl = sl.rename(columns={"category": "sl_label"})
    ahn = pd.read_csv(ROOT / "outputs/analysis/taxonomy/sl32_vs_ahn_mapping.csv")
    ahn = ahn.rename(columns={"ssv2_class_id": "class_id"})[["class_id", "ahn_label"]]
    return sl.merge(ahn, on="class_id", how="left")  # ahn_label NaN for 38/97/160


def build_b1_four_bucket_fractions() -> pd.DataFrame:
    """Class-weighted four-bucket (noise/sign_flip/decrease/increase) fractions,
    both strata, both taxonomies, both backbones (brief §4 B1 / open item 10).
    Class-weighted: mean-per-class first, then equal-weighted average across
    classes in the stratum - not a raw clip pool average.
    """
    labels = _load_taxonomy_labels()
    rows = []
    for backbone, relpath in [("VM", "outputs/analysis/shuffle_reduction_composition/ssv2_vm_clip_shuffle_disruption.csv"),
                               ("TF", "outputs/analysis/shuffle_reduction_composition/ssv2_tf_clip_shuffle_disruption.csv")]:
        path = ROOT / relpath
        clips = pd.read_csv(path)
        per_class = clips.groupby("class_id")[BUCKET_COLS].mean().reset_index()
        per_class["n_clips"] = clips.groupby("class_id").size().values
        per_class = per_class.merge(labels, on="class_id", how="left")

        for taxonomy, label_col in [("SL", "sl_label"), ("Ahn", "ahn_label")]:
            strata = per_class[label_col].dropna().unique() if taxonomy == "Ahn" else per_class[label_col].unique()
            for stratum in strata:
                sub = per_class[per_class[label_col] == stratum]
                rows.append({
                    "backbone": backbone, "taxonomy": taxonomy, "stratum": stratum,
                    "n_classes": len(sub), "n_clips": int(sub["n_clips"].sum()),
                    **{c: round(sub[c].mean(), 4) for c in BUCKET_COLS},
                    "pool": "SL-33" if backbone == "VM" else "SL-31",
                    "unit": "clip", "weighting": "class-weighted", "condition": "R vs shuffle",
                    "source_file": str(path), "source_mtime": mtime_date(path),
                    "status": status_for(path, fix_relevant=(backbone == "VM")),
                })
        # classes present in the clip pool but absent from the 32-class Ahn mapping
        unmapped = per_class[per_class["ahn_label"].isna()]
        if len(unmapped):
            rows.append({
                "backbone": backbone, "taxonomy": "Ahn", "stratum": "not_in_ahn32_mapping",
                "n_classes": len(unmapped), "n_clips": int(unmapped["n_clips"].sum()),
                **{c: round(unmapped[c].mean(), 4) for c in BUCKET_COLS},
                "pool": "SL-33" if backbone == "VM" else "SL-31",
                "unit": "clip", "weighting": "class-weighted", "condition": "R vs shuffle",
                "source_file": str(path), "source_mtime": mtime_date(path),
                "status": status_for(path, fix_relevant=(backbone == "VM")),
            })
    return pd.DataFrame(rows, columns=B1_COLUMNS)


# ============================ TASK 2: B2 logistic regression ORs ================

B2_COLUMNS = [
    "backbone", "version", "term", "coef", "odds_ratio", "se", "z", "p",
    "n", "base_rate", "converged", "pool", "unit", "weighting", "condition",
    "source_file", "source_mtime", "status",
]


def build_b2_logistic_regression_ors() -> pd.DataFrame:
    """Bucket ORs, noise as reference, per backbone (brief §4 B2) - 'pooled'
    version alongside 'class_fe' (class-fixed-effects / class-demeaned - the
    within/between decomposition the brief asks be shown adjacent). Both
    already computed by clip_shuffle_disruption.py's fit_logit/fit_logit_class_fe
    - not re-derived here. odds_ratio = exp(coef), added since the brief asks
    for ORs and the source file only carries the raw logit coefficient.
    """
    rows = []
    for backbone, relpath in [("VM", "outputs/analysis/shuffle_reduction_composition/ssv2_vm_logit_coefficients.csv"),
                               ("TF", "outputs/analysis/shuffle_reduction_composition/ssv2_tf_logit_coefficients.csv")]:
        path = ROOT / relpath
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "backbone": backbone, "version": r["version"], "term": r["term"],
                "coef": r["coef"], "odds_ratio": round(float(np.exp(r["coef"])), 4),
                "se": r["se"], "z": r["z"], "p": r["p"], "n": r["n"],
                "base_rate": r["base_rate"], "converged": r["converged"],
                "pool": "SL-33" if backbone == "VM" else "SL-31", "unit": "clip",
                "weighting": "n/a", "condition": "R vs shuffle (P(correct|shuffle))",
                "source_file": str(path), "source_mtime": mtime_date(path),
                "status": status_for(path, fix_relevant=(backbone == "VM")),
            })
    return pd.DataFrame(rows, columns=B2_COLUMNS)


# ============================ TASK 2: B3 K400 sign-flip contrast ================

B3_COLUMNS = [
    "backbone", "metric", "value", "n", "pool", "unit", "weighting",
    "condition", "source_file", "source_mtime", "status",
]


def build_b3_k400_sign_flip() -> pd.DataFrame:
    """K400 sign-flip contrast, VM only (brief §4 B3). Per-clip mean/median/
    %clips-with->=1-flip are MISSING, not computed: kinetics400_vm_full_detail.parquet
    has no clip_id column at all. Traced why - full_detail_table.py (which adds
    clip_id) only has CONFIGS entries for ssv2_vm/ssv2_tf (clip_shuffle_disruption.py's
    own docstring: 'VM-SSv2 and TF-SSv2 only'); K400's full_detail file was
    produced by a different, incomplete route that never got a clip-id join.
    Its 'bucket' column does carry the same 4-way taxonomy (noise/sign_flip/
    decrease/increase) though, so an instance-level (not per-clip) sign-flip
    fraction is reported alongside as a clearly-separate, lesser statistic.
    """
    path = ROOT / "outputs/analysis/shuffle_reduction_composition/kinetics400_vm_full_detail.parquet"
    df = pd.read_parquet(path)
    n = len(df)
    common = {"backbone": "VM", "pool": "SL-64", "unit": "clip", "weighting": "n/a",
              "condition": "R vs shuffle", "source_file": str(path),
              "source_mtime": mtime_date(path), "status": status_for(path)}
    rows = [
        {**common, "metric": "mean_frac_sign_flip_per_clip", "value": None, "n": None, "status": "MISSING"},
        {**common, "metric": "median_frac_sign_flip_per_clip", "value": None, "n": None, "status": "MISSING"},
        {**common, "metric": "pct_clips_with_ge1_flip", "value": None, "n": None, "status": "MISSING"},
        {**common, "metric": "instance_level_sign_flip_fraction (NOT per-clip - no clip_id in source)",
         "value": round((df["bucket"] == "sign_flip").mean(), 4), "n": n, "unit": "(feature,clip)"},
    ]
    return pd.DataFrame(rows, columns=B3_COLUMNS)


# =========================== TASK 2: B4 top-10 mass coverage ====================

B4_COLUMNS = [
    "backbone", "weighting", "n_clips", "n_classes", "mean_coverage",
    "median_coverage", "pool", "unit", "condition", "source_file_detail",
    "source_file_mass", "source_mtime", "status",
]

# NOTE on naming: the brief's §4 B4 calls this "top-12 coverage", but the
# actual source (ssv2_*_full_detail.parquet) stores each clip's own top-N by
# |signed_R| with TOP_N=10 (clip_shuffle_disruption.py's constant, checked
# directly) - not a fixed global 12-feature set (that's TOP12, a different,
# unrelated concept from TF's sign-flip ablation targets). Built as "top-10
# coverage" to match what's actually on disk; flagged, not silently renamed
# to match the brief's "12".
B4_SOURCES = [
    ("VM", "outputs/analysis/shuffle_reduction_composition/ssv2_vm_full_detail.parquet",
     "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_l7_job7ep_k64.parquet"),
    ("TF", "outputs/analysis/shuffle_reduction_composition/ssv2_tf_full_detail.parquet",
     "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet"),
]


def build_b4_top10_coverage() -> pd.DataFrame:
    """Concentration - top-10 (see naming note above) coverage, instance-weighted
    and class-weighted as separate declared rows (brief §4 B4). coverage =
    clip's stored top-10 |signed_R| sum / that clip's total_abs_R (whole-
    dictionary mass), joined by clip_id from the matching mass-delta parquet.
    """
    rows = []
    for backbone, detail_relpath, mass_relpath in B4_SOURCES:
        detail_path, mass_path = ROOT / detail_relpath, ROOT / mass_relpath
        detail = pd.read_parquet(detail_path)
        mass = pd.read_parquet(mass_path)[["clip_id", "class_id", "total_abs_R"]]

        top10_mass = detail.groupby("clip_id")["signed_R"].apply(lambda s: s.abs().sum()).reset_index(name="top10_abs_mass")
        merged = top10_mass.merge(mass, on="clip_id", how="inner")
        merged["coverage"] = merged["top10_abs_mass"] / merged["total_abs_R"]

        common = {"backbone": backbone, "pool": "SL-35", "unit": "clip",
                  "condition": "R", "source_file_detail": str(detail_path),
                  "source_file_mass": str(mass_path), "source_mtime": mtime_date(detail_path),
                  "status": status_for(detail_path, fix_relevant=(backbone == "VM"))}
        rows.append({
            **common, "weighting": "instance-weighted", "n_clips": len(merged),
            "n_classes": merged["class_id"].nunique(),
            "mean_coverage": round(merged["coverage"].mean(), 4),
            "median_coverage": round(merged["coverage"].median(), 4),
        })
        per_class = merged.groupby("class_id")["coverage"].mean()
        rows.append({
            **common, "weighting": "class-weighted", "n_clips": len(merged),
            "n_classes": len(per_class),
            "mean_coverage": round(per_class.mean(), 4),
            "median_coverage": round(per_class.median(), 4),
        })
    return pd.DataFrame(rows, columns=B4_COLUMNS)


# ============================== TASK 2: B5 TOP12 ablation ========================

B5_COLUMNS = [
    "ablation_target", "condition", "stratum", "n_clips", "flip_rate",
    "mean_logit_damage", "median_logit_damage", "mean_baseline_logit",
    "median_baseline_logit", "pool", "unit", "weighting", "source_file",
    "source_mtime", "status",
]


def build_b5_top12_ablation() -> pd.DataFrame:
    """TOP12 flip rate + logit damage, static/temporal split, medians primary
    with means alongside, unablated baseline logits included as the confound
    check (brief §4 B5). Brief names tf_signflip_flip_rate_logit_damage.csv,
    which has neither the taxonomy split nor medians nor baseline logits -
    recomputed from the per-clip source it's itself aggregated from
    (tf_signflip_ablation_results_long.parquet), cross-checked exactly against
    the named file's TOP12/R aggregate (0.748632 flip rate, 9.490705 mean
    damage - both match to 6 decimals) before trusting the stratified version.
    TF has no C1 (not R and C1 as the brief's condition axis says - TF genuinely
    has R and C only, consistent with P1/A5's same finding).
    """
    path = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_results_long.parquet"
    df = pd.read_parquet(path)
    top12 = df[df["ablation_target"] == "TOP12"]

    rows = []
    for cond in ("R", "C"):
        by_cond = top12[top12["perturbation_condition"] == cond]
        for stratum, sub in [("static", by_cond[by_cond["sl_label"] == "static"]),
                              ("temporal", by_cond[by_cond["sl_label"] == "temporal"]),
                              ("overall", by_cond)]:
            flipped = ~sub["correct_ablated"]
            rows.append({
                "ablation_target": "TOP12", "condition": cond, "stratum": stratum,
                "n_clips": len(sub), "flip_rate": round(flipped.mean(), 4),
                "mean_logit_damage": round(sub["delta"].mean(), 4),
                "median_logit_damage": round(sub["delta"].median(), 4),
                "mean_baseline_logit": round(sub["baseline_logit"].mean(), 4),
                "median_baseline_logit": round(sub["baseline_logit"].median(), 4),
                "pool": "SL-35", "unit": "clip", "weighting": "n/a",
                "source_file": str(path), "source_mtime": mtime_date(path),
                "status": status_for(path, fix_relevant=False),
            })
    rows.append({
        "ablation_target": "TOP12", "condition": "C1", "stratum": "n/a",
        "n_clips": None, "flip_rate": None, "mean_logit_damage": None,
        "median_logit_damage": None, "mean_baseline_logit": None,
        "median_baseline_logit": None, "pool": "SL-35", "unit": "clip",
        "weighting": "n/a", "source_file": None, "source_mtime": None,
        "status": "MISSING",
    })
    return pd.DataFrame(rows, columns=B5_COLUMNS)


# ================================ TASK 2: D1 C vs C1 vs TF C =====================

D1_COLUMNS = [
    "backbone", "condition", "pool", "n_classes", "n_clips", "correct",
    "top1_accuracy", "unit", "weighting", "source_file", "source_mtime", "status",
]

# Two pools, per Tom's call (brief §4 D1, open item 1) - report both, pick
# neither. Identified empirically: TF's C accuracy moves 4.95pp between the
# full-174-class ungated SSv2 val and the SL-35 filter, matching the brief's
# cited "5.0pp" figure to within rounding - confirms this is the intended pair.
D1_SOURCES = [
    ("VM", "C", "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_C.csv", True),
    ("VM", "C1", "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_C1.csv", True),
    ("TF", "C", "outputs/stage1_class_selection_TF/per_class_accuracy_TF_C.csv", False),
]


def build_d1_c_vs_c1_vs_tfc() -> pd.DataFrame:
    """C vs C1 vs TF C top-1, both pools labeled, neither picked (brief §4 D1)."""
    sl35 = load_sl35_class_ids()
    rows = []
    for backbone, cond, relpath, fix_relevant in D1_SOURCES:
        path = ROOT / relpath
        df = pd.read_csv(path)
        for pool_label, sub in [("SSv2 val (full 174-class, ungated)", df),
                                 ("SL-35", df[df["class_id"].isin(sl35)])]:
            correct, total = int(sub["correct"].sum()), int(sub["total"].sum())
            rows.append({
                "backbone": backbone, "condition": cond, "pool": pool_label,
                "n_classes": len(sub), "n_clips": total, "correct": correct,
                "top1_accuracy": round(correct / total, 4), "unit": "clip",
                "weighting": "clip-weighted", "source_file": str(path),
                "source_mtime": mtime_date(path),
                "status": status_for(path, fix_relevant=fix_relevant),
            })
    return pd.DataFrame(rows, columns=D1_COLUMNS)


# ========================= TASK 2: B6 taxonomy robustness falsifiers ===========

B6_FALSIFIER_COLUMNS = [
    "falsifier", "taxonomy", "stratum", "vm_value", "tf_value", "vm_tf_ratio",
    "pool", "unit", "weighting", "source_file", "source_mtime", "status",
]
B6_DIFF_COLUMNS = [
    "config", "condition", "ablation_target", "diff_in_diff",
    "pool", "unit", "weighting", "source_file", "source_mtime", "status",
]

# Transcribed directly from taxonomy_bucket_comparison_120826.md's own finished
# "Step 3" result tables - not recomputed. Pre-registered falsifier (Tom,
# 12/08, logged before results): shape should be unchanged between SL and Ahn
# taxonomies; failure = the VM/TF ratio itself moving. Both swings reported in
# the brief (0.0026, 0.0008) are transcribed here exactly as published.
B6_NOISE_RATIO_FALSIFIER = [
    ("SL", "static", 0.0436, 0.1414, 0.3082), ("SL", "temporal", 0.0309, 0.1109, 0.2786),
    ("Ahn", "static", 0.0422, 0.1373, 0.3073), ("Ahn", "temporal", 0.0309, 0.1099, 0.2812),
    ("Ahn", "unassigned", 0.0388, 0.1906, 0.2036),
]
B6_TOP12_COVERAGE_FALSIFIER = [
    ("SL", "overall", 0.2645, 0.8255, 0.3204), ("Ahn", "overall", 0.2652, 0.8255, 0.3212),
]


def build_b6_taxonomy_falsifiers() -> pd.DataFrame:
    """Both pre-registered falsifiers (brief §4 B6), transcribed from the
    finished report, not recomputed - the source md's own numbers are the
    result of record, and it already includes a self-check (TF top-12 static
    share = 82.546%, matching a separately-cited figure exactly)."""
    md_path = ROOT / "outputs/analysis/taxonomy/taxonomy_bucket_comparison_120826.md"
    common = {"pool": "SL/Ahn (VM: SL-33/Ahn-32-in-33; TF: SL-31/Ahn-32-in-31)",
              "unit": "class", "weighting": "class-weighted",
              "source_file": str(md_path), "source_mtime": mtime_date(md_path),
              "status": status_for(md_path)}
    rows = [{"falsifier": "vm_tf_noise_ratio", "taxonomy": tax, "stratum": strat,
             "vm_value": vm, "tf_value": tf, "vm_tf_ratio": ratio, **common}
            for tax, strat, vm, tf, ratio in B6_NOISE_RATIO_FALSIFIER]
    rows += [{"falsifier": "vm_tf_top12_sign_flip_coverage_ratio", "taxonomy": tax,
              "stratum": strat, "vm_value": vm, "tf_value": tf, "vm_tf_ratio": ratio,
              **{**common, "weighting": "instance-weighted"}}
             for tax, strat, vm, tf, ratio in B6_TOP12_COVERAGE_FALSIFIER]
    rows.append({"falsifier": "max_swing_noise_ratio (Ahn-SL)", "taxonomy": "both",
                 "stratum": "n/a", "vm_value": None, "tf_value": None, "vm_tf_ratio": 0.0026, **common})
    rows.append({"falsifier": "swing_top12_coverage_ratio (Ahn-SL)", "taxonomy": "both",
                 "stratum": "n/a", "vm_value": None, "tf_value": None, "vm_tf_ratio": 0.0008,
                 **{**common, "weighting": "instance-weighted"}})
    return pd.DataFrame(rows, columns=B6_FALSIFIER_COLUMNS)


B6_DIFF_SOURCES = [
    ("VM/SSv2/L5/x8k64", "outputs/analysis/scaffold_ablation/ablation_diff_in_diff_l5_job7ep_k64.csv"),
    ("VM/SSv2/L7/x8k64", "outputs/analysis/scaffold_ablation/ablation_diff_in_diff_l7_job7ep_k64.csv"),
    ("VM/K400/L5/x8k64", "outputs/analysis/scaffold_ablation/ablation_diff_in_diff_kinetics400_l5_job7ep_k64.csv"),
    ("VM/K400/L7/x8k64", "outputs/analysis/scaffold_ablation/ablation_diff_in_diff_kinetics400_l7_job7ep_k64.csv"),
]


def build_b6_diff_in_diff() -> pd.DataFrame:
    """The diff-in-diff table alongside the falsifiers (brief §4 B6) - the
    per-config ablation diff-in-diff files, current (job7ep) only; the
    unsuffixed/clean7 legacy versions excluded, same reasoning as A4."""
    rows = []
    for config, relpath in B6_DIFF_SOURCES:
        path = ROOT / relpath
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            rows.append({
                "config": config, "condition": r["perturbation_condition"],
                "ablation_target": r["ablation_target"], "diff_in_diff": r["diff_in_diff"],
                "pool": "SL-35" if "SSv2" in config else "SL-64", "unit": "(feature,clip)",
                "weighting": "n/a", "source_file": str(path), "source_mtime": mtime_date(path),
                "status": status_for(path),
            })
    return pd.DataFrame(rows, columns=B6_DIFF_COLUMNS)


# ==================== TASK 2 (ad hoc): B7 ablated x sign-flip overlap ===========

B7_COLUMNS = [
    "backbone", "feature_idx", "ablation_role", "in_signflip_pool",
    "position", "position_type", "position_metric",
    "dfa_concentration_R", "concentration_metric",
    "pool", "unit", "weighting", "condition", "source_file", "source_mtime", "status",
]


def build_b7_ablated_signflip_overlap() -> pd.DataFrame:
    """Do the ablated scaffold features live in the shuffle sign-flip bucket
    too? Plus each feature's locked tubelet/frame position and DFA
    concentration (user, 26/08) - both backbones' L7 ablation set, matched by
    feature_id against ssv2_{vm,tf}_full_detail.parquet's sign_flip bucket.
    VM: all 4 canonical ablated features overlap (605-feature pool). TF: all
    12 TOP12 features overlap (48-feature pool). Position and concentration
    for VM read directly from L7_x8k64_VM.csv's dfa_position/dfa_share_R
    (already computed by the pipeline). TF has no equivalent collapsed file
    (never run through scaffold_selection_consolidated.py - see
    DISCREPANCIES.md finding 3), so both are recomputed here using the
    pipeline's own exact methods (collapse_to_feature's `_modal` for position,
    its min-across-classes rule for concentration/share) applied only to the
    12 ablated features, not a new method.
    """
    vm_fd = pd.read_parquet(ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_vm_full_detail.parquet")
    vm_signflip = set(vm_fd[vm_fd["bucket"] == "sign_flip"]["feature_id"].unique())
    tf_fd = pd.read_parquet(ROOT / "outputs/analysis/shuffle_reduction_composition/ssv2_tf_full_detail.parquet")
    tf_signflip = set(tf_fd[tf_fd["bucket"] == "sign_flip"]["feature_id"].unique())

    vm_path = ROOT / "outputs/analysis/scaffold_selection/L7_x8k64_VM.csv"
    vm = pd.read_csv(vm_path)
    rows = []
    for _, r in vm.iterrows():
        if r["status"] not in ("member", "near_miss"):
            continue
        is_ablated = r["feature_idx"] in {5165, 6021, 6032, 3347}
        if not is_ablated:
            continue
        rows.append({
            "backbone": "VM", "feature_idx": int(r["feature_idx"]), "ablation_role": r["status"],
            "in_signflip_pool": r["feature_idx"] in vm_signflip, "position": int(r["dfa_position"]),
            "position_type": "tubelet", "position_metric": "dfa_position (L7_x8k64_VM.csv, pipeline-computed)",
            "dfa_concentration_R": r["dfa_share_R"],
            "concentration_metric": "dfa_share_R (L7_x8k64_VM.csv, pipeline-computed: min per-clip share across 32 classes)",
            "pool": "SL-35", "unit": "feature", "weighting": "n/a", "condition": "R",
            "source_file": str(vm_path), "source_mtime": mtime_date(vm_path), "status": status_for(vm_path),
        })

    tf_targets_path = ROOT / "outputs/analysis/scaffold_ablation/tf_signflip_ablation_targets.json"
    with open(tf_targets_path) as f:
        tf_ablated = json.load(f)["TOP12"]
    tf_pos_path = ROOT / "outputs/analysis/dfa_per_tubelet_mass/position_lock_scores_timesformer_l7.csv"
    tf_pos_df = pd.read_csv(tf_pos_path)
    sub = tf_pos_df[tf_pos_df["feature_idx"].isin(tf_ablated)]
    modal_pos = sub.groupby("feature_idx")["mode_frame_R"].apply(
        lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None)
    concentration = sub.groupby("feature_idx")["mean_per_clip_share_R"].min()  # matches collapse_to_feature's share rule
    for feat in tf_ablated:
        rows.append({
            "backbone": "TF", "feature_idx": feat, "ablation_role": "TOP12",
            "in_signflip_pool": feat in tf_signflip,
            "position": int(modal_pos.get(feat)) if feat in modal_pos.index and pd.notna(modal_pos.get(feat)) else None,
            "position_type": "frame",
            "position_metric": "modal mode_frame_R across 32 classes (recomputed, matching collapse_to_feature's _modal method)",
            "dfa_concentration_R": round(concentration.get(feat), 4) if feat in concentration.index else None,
            "concentration_metric": "min per-clip share_R across 32 classes (recomputed, matching collapse_to_feature's share rule)",
            "pool": "SL-32", "unit": "feature", "weighting": "n/a", "condition": "R",
            "source_file": str(tf_pos_path), "source_mtime": mtime_date(tf_pos_path),
            "status": status_for(tf_pos_path, fix_relevant=False),
        })
    return pd.DataFrame(rows, columns=B7_COLUMNS)


# ================================ TASK 2: appendix ================================

X1_COLUMNS = ["backbone", "condition", "class_id", "template", "correct",
              "total", "accuracy", "pool", "unit", "weighting", "source_file", "source_mtime", "status"]


def build_x1_ssv2_per_class() -> pd.DataFrame:
    """Per-class breakdown, SSv2, both backbones, all conditions, SL-35 ungated
    (brief §4 X1) - same sources as P1, unaggregated."""
    sl35 = load_sl35_class_ids()
    rows = []
    for backbone, cond, relpath, fix_relevant in P1_SOURCES:
        path = ROOT / relpath
        df = pd.read_csv(path)
        sub = df[df["class_id"].isin(sl35)]
        for _, r in sub.iterrows():
            rows.append({
                "backbone": backbone, "condition": cond, "class_id": r["class_id"],
                "template": r["template"], "correct": r["correct"], "total": r["total"],
                "accuracy": r["accuracy"], "pool": "SL-35", "unit": "clip",
                "weighting": "n/a (per-class row)",
                "source_file": str(path), "source_mtime": mtime_date(path),
                "status": status_for(path, fix_relevant=fix_relevant),
            })
    return pd.DataFrame(rows, columns=X1_COLUMNS)


X2_COLUMNS = ["backbone", "condition", "class_id", "template", "correct",
              "total", "accuracy", "pool", "unit", "weighting", "source_file", "source_mtime", "status"]


def build_x2_k400_per_class() -> pd.DataFrame:
    """Per-class breakdown, K400, VM, SL-64 (brief §4 X2) - same sources as
    P2, unaggregated. Subject to the same P2 contradiction (B/C MISSING)."""
    sl64 = load_sl64_class_ids()
    rows = []
    for cond, relpath in P2_SOURCES:
        if relpath is None:
            rows.append({"backbone": "VM", "condition": cond, "class_id": None,
                         "template": None, "correct": None, "total": None, "accuracy": None,
                         "pool": "SL-64", "unit": "clip", "weighting": "n/a (per-class row)",
                         "source_file": None, "source_mtime": None, "status": "MISSING"})
            continue
        path = ROOT / relpath
        df = pd.read_csv(path)
        sub = df[df["class_id"].isin(sl64)]
        for _, r in sub.iterrows():
            rows.append({
                "backbone": "VM", "condition": cond, "class_id": r["class_id"],
                "template": r["template"], "correct": r["correct"], "total": r["total"],
                "accuracy": r["accuracy"], "pool": "SL-64", "unit": "clip",
                "weighting": "n/a (per-class row)",
                "source_file": str(path), "source_mtime": mtime_date(path), "status": status_for(path),
            })
    return pd.DataFrame(rows, columns=X2_COLUMNS)


def build_x3_spliced_accuracy_full() -> pd.DataFrame:
    """Spliced accuracy, full, per layer/config/epoch (brief §4 X3). Epoch is
    MISSING throughout - the on-disk per-class spliced accuracy CSVs
    (outputs/spliced_accuracy_vm/*_best.csv) only carry the '_best' checkpoint
    per config, not a per-epoch sweep; P3's sweep parquet is likewise best-only.
    Reports the same per-class detail P3 aggregates, per config, labeled
    'best' rather than a specific epoch number.
    """
    rows = []
    for f in sorted((ROOT / "outputs/spliced_accuracy_vm").glob("spliced_accuracy_*_best.csv")):
        df = pd.read_csv(f)
        df["checkpoint_epoch"] = "best (per-epoch sweep not on disk)"
        df["pool"] = "held-out SAE split (not SL-gated)"
        df["unit"] = "clip"
        df["weighting"] = "n/a (per-class row)"
        df["source_file"] = str(f)
        df["source_mtime"] = mtime_date(f)
        df["status"] = status_for(f)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


X5_COLUMNS = ["feature_idx", "member_status", "share_R_dfa", "consistency_R_dfa",
              "share_C1_dfa", "pool", "unit", "weighting", "source_file", "source_mtime", "status"]


def build_x5_l7_near_miss_detail() -> pd.DataFrame:
    """L7 near-miss detail feeding A7's natural-break argument (brief §4 X5) -
    the same L7_x8k64_VM.csv rows A7 already surfaces, as their own appendix
    table rather than mixed into A7's percentile rows.
    """
    path = ROOT / "outputs/analysis/scaffold_selection/L7_x8k64_VM.csv"
    df = pd.read_csv(path)
    out = df[["feature_idx", "status", "dfa_share_R", "dfa_consistency_R", "dfa_share_C1"]].copy()
    out.columns = ["feature_idx", "member_status", "share_R_dfa", "consistency_R_dfa", "share_C1_dfa"]
    out["pool"] = "SL-35"
    out["unit"] = "feature"
    out["weighting"] = "n/a"
    out["source_file"] = str(path)
    out["source_mtime"] = mtime_date(path)
    out["status"] = status_for(path)
    return out[X5_COLUMNS]


# ==================== TASK 2 (ad hoc): X6 VM position-locked member detail ======

X6_COLUMNS = [
    "config", "backbone", "layer", "sae_k", "dataset", "feature_idx",
    "share_R_dfa", "share_C1_dfa", "share_A_dfa", "consistency_R_dfa",
    "consistency_C1_dfa", "consistency_A_dfa", "position_dfa",
    "share_R_z", "consistency_R_z", "position_z",
    "top5_classes_dfa_r", "bottom5_classes_dfa_r",
    "pool", "unit", "weighting", "condition", "source_file", "source_mtime", "status",
]


def build_x6_vm_position_locked_members() -> pd.DataFrame:
    """Every VM position-locked (status == member) feature's own feature_idx
    plus its full per-feature statistics, across all 11 VM configs - not just
    L7's near-miss detail (X5) or the small ablated subset (B7). User (26/08):
    the position-locked features need their idx reported alongside their other
    stats, not just the A1 member counts.
    """
    rows = []
    for name, layer, dataset in VM_SSV2_CONFIGS + VM_K400_CONFIGS:
        path = ROOT / "outputs/analysis/scaffold_selection" / f"{name}.csv"
        if not path.exists():
            rows.append({"config": name, "backbone": "VM", "layer": layer, "dataset": dataset,
                         "feature_idx": None, "share_R_dfa": None, "share_C1_dfa": None,
                         "share_A_dfa": None, "consistency_R_dfa": None, "consistency_C1_dfa": None,
                         "consistency_A_dfa": None, "position_dfa": None, "share_R_z": None,
                         "consistency_R_z": None, "position_z": None, "top5_classes_dfa_r": None,
                         "bottom5_classes_dfa_r": None, "pool": "SL-35" if dataset == "ssv2" else "SL-64",
                         "unit": "feature", "weighting": "n/a", "condition": "R",
                         "source_file": None, "source_mtime": None, "status": "MISSING"})
            continue
        df = pd.read_csv(path)
        members = df[df["status"] == "member"]
        common = {"config": name, "backbone": "VM", "layer": layer, "dataset": dataset,
                  "pool": "SL-35" if dataset == "ssv2" else "SL-64", "unit": "feature",
                  "weighting": "n/a", "condition": "R", "source_file": str(path),
                  "source_mtime": mtime_date(path), "status": status_for(path)}
        if members.empty:
            rows.append({**common, "feature_idx": None, "share_R_dfa": None, "share_C1_dfa": None,
                        "share_A_dfa": None, "consistency_R_dfa": None, "consistency_C1_dfa": None,
                        "consistency_A_dfa": None, "position_dfa": None, "share_R_z": None,
                        "consistency_R_z": None, "position_z": None, "top5_classes_dfa_r": None,
                        "bottom5_classes_dfa_r": None})
            continue
        for _, r in members.iterrows():
            rows.append({
                **common, "feature_idx": int(r["feature_idx"]),
                "share_R_dfa": r["dfa_share_R"], "share_C1_dfa": r["dfa_share_C1"],
                "share_A_dfa": r["dfa_share_A"], "consistency_R_dfa": r["dfa_consistency_R"],
                "consistency_C1_dfa": r["dfa_consistency_C1"], "consistency_A_dfa": r["dfa_consistency_A"],
                "position_dfa": r["dfa_position"], "share_R_z": r["z_share_R"],
                "consistency_R_z": r["z_consistency_R"], "position_z": r["z_position"],
                "top5_classes_dfa_r": r["top5_classes_dfa_r"], "bottom5_classes_dfa_r": r["bottom5_classes_dfa_r"],
            })
    return pd.DataFrame(rows, columns=X6_COLUMNS)


X4_COLUMNS = ["config", "metric", "value", "pool", "unit", "weighting", "source_file", "source_mtime", "status"]
X4_METRICS = ["chosen_epoch", "r_squared", "dead_feature_fraction", "l0", "spliced_accuracy"]


def build_x4_sae_selection_outcomes() -> pd.DataFrame:
    """Per-SAE selection outcomes (brief §4 X4) - entirely MISSING this pass.
    Brief's own note says 'recoverable from logs / W&B'; checked directly for
    a local path: no train_sae*.out logs exist anywhere in the repo, and W&B
    isn't reachable from this session. Nothing to extract without either -
    reported MISSING per config/metric rather than left off the manifest
    silently, since the brief calls this table out as an explicit defence
    against the dictionary-artifact objection.
    """
    configs = [
        "L3_x8k64_VM", "L5_x8k64_VM", "L7_x8k64_VM", "L5_x16k128_VM", "L7_x16k128_VM",
        "L9_x8k64_VM", "L9_x16k128_VM", "L5_x8k64_TF", "L7_x8k64_TF", "L9_x8k64_TF",
        "L3_x8k64_VM_K400", "L5_x8k64_VM_K400", "L7_x8k64_VM_K400", "L9_x8k64_VM_K400",
    ]
    rows = []
    for c in configs:
        for metric in X4_METRICS:
            rows.append({"config": c, "metric": metric, "value": None,
                         "pool": None, "unit": "n/a", "weighting": "n/a",
                         "source_file": None, "source_mtime": None, "status": "MISSING"})
    return pd.DataFrame(rows, columns=X4_COLUMNS)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = build_t0_pool_census()
    t0.to_csv(OUT_DIR / "T0_pool_census.csv", index=False)

    p1 = build_p1_ssv2_condition_accuracy()
    p1.to_csv(OUT_DIR / "P1_ssv2_condition_accuracy.csv", index=False)

    p2 = build_p2_k400_condition_accuracy()
    p2.to_csv(OUT_DIR / "P2_k400_condition_accuracy.csv", index=False)

    p3 = build_p3_spliced_accuracy()
    p3.to_csv(OUT_DIR / "P3_spliced_accuracy.csv", index=False)

    a1 = build_a1_scaffold_member_counts()
    a1.to_csv(OUT_DIR / "A1_scaffold_member_counts.csv", index=False)

    a2 = build_a2_mass_concentration_bootstrap()
    a2.to_csv(OUT_DIR / "A2_mass_concentration_bootstrap.csv", index=False)

    a3 = build_a3_ceiling_occupancy()
    a3.to_csv(OUT_DIR / "A3_ceiling_occupancy.csv", index=False)

    a4 = build_a4_ablation_additivity()
    a4.to_csv(OUT_DIR / "A4_ablation_additivity.csv", index=False)

    a5 = build_a5_temporal_static_ablation()
    a5.to_csv(OUT_DIR / "A5_temporal_static_ablation.csv", index=False)

    a6 = build_a6_per_class_alignment()
    a6.to_csv(OUT_DIR / "A6_per_class_alignment.csv", index=False)

    a7 = build_a7_near_miss_distribution()
    a7.to_csv(OUT_DIR / "A7_near_miss_distribution.csv", index=False)

    b1 = build_b1_four_bucket_fractions()
    b1.to_csv(OUT_DIR / "B1_four_bucket_fractions.csv", index=False)

    b2 = build_b2_logistic_regression_ors()
    b2.to_csv(OUT_DIR / "B2_logistic_regression_ors.csv", index=False)

    b3 = build_b3_k400_sign_flip()
    b3.to_csv(OUT_DIR / "B3_k400_sign_flip.csv", index=False)

    b4 = build_b4_top10_coverage()
    b4.to_csv(OUT_DIR / "B4_top10_coverage.csv", index=False)

    b5 = build_b5_top12_ablation()
    b5.to_csv(OUT_DIR / "B5_top12_ablation.csv", index=False)

    b6f = build_b6_taxonomy_falsifiers()
    b6f.to_csv(OUT_DIR / "B6_taxonomy_falsifiers.csv", index=False)
    b6d = build_b6_diff_in_diff()
    b6d.to_csv(OUT_DIR / "B6_diff_in_diff.csv", index=False)

    b7 = build_b7_ablated_signflip_overlap()
    b7.to_csv(OUT_DIR / "B7_ablated_signflip_overlap.csv", index=False)

    d1 = build_d1_c_vs_c1_vs_tfc()
    d1.to_csv(OUT_DIR / "D1_c_vs_c1_vs_tfc.csv", index=False)

    x1 = build_x1_ssv2_per_class()
    x1.to_csv(OUT_DIR / "X1_ssv2_per_class.csv", index=False)

    x2 = build_x2_k400_per_class()
    x2.to_csv(OUT_DIR / "X2_k400_per_class.csv", index=False)

    x3 = build_x3_spliced_accuracy_full()
    x3.to_csv(OUT_DIR / "X3_spliced_accuracy_full.csv", index=False)

    x4 = build_x4_sae_selection_outcomes()
    x4.to_csv(OUT_DIR / "X4_sae_selection_outcomes.csv", index=False)

    x5 = build_x5_l7_near_miss_detail()
    x5.to_csv(OUT_DIR / "X5_l7_near_miss_detail.csv", index=False)

    x6 = build_x6_vm_position_locked_members()
    x6.to_csv(OUT_DIR / "X6_vm_position_locked_members.csv", index=False)

    write_manifest({
        "T0": ("T0_pool_census.csv", t0),
        "P1": ("P1_ssv2_condition_accuracy.csv", p1),
        "P2": ("P2_k400_condition_accuracy.csv", p2),
        "P3": ("P3_spliced_accuracy.csv", p3),
        "A1": ("A1_scaffold_member_counts.csv", a1),
        "A2": ("A2_mass_concentration_bootstrap.csv", a2),
        "A3": ("A3_ceiling_occupancy.csv", a3),
        "A4": ("A4_ablation_additivity.csv", a4),
        "A5": ("A5_temporal_static_ablation.csv", a5),
        "A6": ("A6_per_class_alignment.csv", a6),
        "A7": ("A7_near_miss_distribution.csv", a7),
        "B1": ("B1_four_bucket_fractions.csv", b1),
        "B2": ("B2_logistic_regression_ors.csv", b2),
        "B3": ("B3_k400_sign_flip.csv", b3),
        "B4": ("B4_top10_coverage.csv", b4),
        "B5": ("B5_top12_ablation.csv", b5),
        "B6_falsifiers": ("B6_taxonomy_falsifiers.csv", b6f),
        "B6_diff_in_diff": ("B6_diff_in_diff.csv", b6d),
        "B7": ("B7_ablated_signflip_overlap.csv", b7),
        "D1": ("D1_c_vs_c1_vs_tfc.csv", d1),
        "X1": ("X1_ssv2_per_class.csv", x1),
        "X2": ("X2_k400_per_class.csv", x2),
        "X3": ("X3_spliced_accuracy_full.csv", x3),
        "X4": ("X4_sae_selection_outcomes.csv", x4),
        "X5": ("X5_l7_near_miss_detail.csv", x5),
        "X6": ("X6_vm_position_locked_members.csv", x6),
    })


if __name__ == "__main__":
    main()
