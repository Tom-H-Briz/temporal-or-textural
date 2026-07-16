"""
Consolidated scaffold membership selection — derives feature membership from
scratch against a uniform gate, across all 7 VM/TF configs, from existing
per-clip-derived DFA and z position-lock outputs. No new extraction runs.

Gate (per feature, must hold in EVERY one of the 32 DFA_CLASSES):
  DFA: share_R/C1/A >= 0.90, frac_clips_matching_mode_R/C1/A == 1.0
  z:   share_R      >= 0.90, frac_clips_matching_mode_R      == 1.0
  (z gated on R only — z C1/A columns don't exist for every config on disk;
   confirmed with Tom 15/07/26, applied uniformly rather than as a one-off patch)

Outputs:
  outputs/analysis/scaffold_selection/{config}_all_features.csv  — unfiltered
  outputs/analysis/scaffold_selection/{config}.csv                — member+near_miss
  outputs/analysis/scaffold_selection/scaffold_selection_ceiling_summary.csv

Usage:
    uv run python src/stage3_analysis/scaffold_selection_consolidated.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

DFA_CLASSES = {0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
               41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
               168, 169, 171, 173}

# {config} label is `{layer}_x{expansion}k{k}_{backbone}` — no exact match found
# elsewhere in the codebase (job labels use "64"/"128_16x"), so this is a newly
# introduced naming convention, not matched to an existing one.
CONFIGS = [
    {"name": "L5_x8k64_VM",   "model": "videomae",    "layer": 5, "job_label": "64",      "sae_k": 64},
    {"name": "L7_x8k64_VM",   "model": "videomae",    "layer": 7, "job_label": "64",      "sae_k": 64},
    {"name": "L7_x16k128_VM", "model": "videomae",    "layer": 7, "job_label": "128_16x", "sae_k": 128},
    {"name": "L9_x8k64_VM",   "model": "videomae",    "layer": 9, "job_label": "64",      "sae_k": 64},
    {"name": "L5_x8k64_TF",   "model": "timesformer", "layer": 5, "job_label": None,      "sae_k": None},
    {"name": "L7_x8k64_TF",   "model": "timesformer", "layer": 7, "job_label": None,      "sae_k": None},
    {"name": "L9_x8k64_TF",   "model": "timesformer", "layer": 9, "job_label": None,      "sae_k": None},
]

GATE = {"min_share": 0.90, "exact_frac": 1.0}
N_NEAR_MISS = 20
N_CLASS_PROFILE = 5

CFG = {
    "dfa_dir":        ROOT / "outputs/analysis/dfa_per_tubelet_mass",
    "z_dir":          ROOT / "outputs/analysis/z_position_lock",
    "mass_delta_dir": ROOT / "outputs/analysis/dfa_mass_delta_vm_c1",
    "out_dir":        ROOT / "outputs/analysis/scaffold_selection",
}


def _resolve_scored_csv(base_dir: Path, prefix: str, cfg: dict) -> Path:
    """Locate the per-class score CSV for a config — suffixed pattern preferred,
    falling back to the legacy unsuffixed file only for the original L7/job64/k64
    baseline (matches scaffold_mass_pct.locate_source's precedent)."""
    model, layer = cfg["model"], cfg["layer"]
    if model == "timesformer":
        path = base_dir / f"{prefix}_timesformer_l{layer}.csv"
    else:
        job_label, sae_k = cfg["job_label"], cfg["sae_k"]
        suffixed = base_dir / f"{prefix}_videomae_l{layer}_job{job_label}_k{sae_k}.csv"
        if suffixed.exists():
            return suffixed
        legacy = base_dir / f"{prefix}.csv"
        if (layer, job_label, sae_k) == (7, "64", 64) and legacy.exists():
            path = legacy
        else:
            path = suffixed
    if not path.exists():
        raise FileNotFoundError(f"No score CSV found for {cfg['name']}: {path}")
    return path


def _resolve_mass_delta_parquet(cfg: dict) -> Path:
    """VM only — used for the ceiling check's per-clip signed-vector floor."""
    layer, job_label, sae_k = cfg["layer"], cfg["job_label"], cfg["sae_k"]
    d = CFG["mass_delta_dir"]
    suffixed = d / f"dfa_mass_delta_vm_c1_l{layer}_job{job_label}_k{sae_k}.parquet"
    if suffixed.exists():
        return suffixed
    legacy = d / "dfa_mass_delta_vm_c1.parquet"
    if (layer, job_label, sae_k) == (7, "64", 64) and legacy.exists():
        return legacy
    raise FileNotFoundError(f"No mass-delta parquet found for {cfg['name']}: {suffixed}")


def _shuffle_raw_label(model: str) -> str:
    return "C" if model == "timesformer" else "C1"


def _position_col(model: str) -> str:
    return "mode_frame" if model == "timesformer" else "mode_tubelet"


def load_dfa(cfg: dict) -> pd.DataFrame:
    """Per (class_id, feature_idx) DFA scores, columns normalized to common
    internal names (share_*, consistency_*, position_*) — no model-conditional
    branching past this point, per project convention."""
    path = _resolve_scored_csv(CFG["dfa_dir"], "position_lock_scores", cfg)
    df = pd.read_csv(path)
    shuf, pos_col = _shuffle_raw_label(cfg["model"]), _position_col(cfg["model"])
    out = df[["class_id", "feature_idx", "total_abs_R"]].copy()
    for raw, cond in [("R", "R"), (shuf, "C1"), ("A", "A")]:
        out[f"share_{cond}"] = df[f"mean_per_clip_share_{raw}"]
        out[f"consistency_{cond}"] = df[f"frac_clips_matching_mode_{raw}"]
        out[f"position_{cond}"] = df[f"{pos_col}_{raw}"]
    return out


def load_z(cfg: dict) -> pd.DataFrame:
    """Per (class_id, feature_idx) z scores. Gate uses R only (Tom, 15/07/26) —
    C1/A columns are populated when present (post-refactor CSVs) and left NaN
    otherwise (legacy L7/job64 z CSV), never gated on either way."""
    path = _resolve_scored_csv(CFG["z_dir"], "z_position_lock_scores", cfg)
    df = pd.read_csv(path)
    shuf, pos_col = _shuffle_raw_label(cfg["model"]), _position_col(cfg["model"])
    out = df[["class_id", "feature_idx"]].copy()
    for raw, cond in [("R", "R"), (shuf, "C1"), ("A", "A")]:
        share_col, frac_col, p_col = f"mean_per_clip_share_{raw}", f"frac_clips_matching_mode_{raw}", f"{pos_col}_{raw}"
        out[f"share_{cond}"] = df[share_col] if share_col in df.columns else np.nan
        out[f"consistency_{cond}"] = df[frac_col] if frac_col in df.columns else np.nan
        out[f"position_{cond}"] = df[p_col] if p_col in df.columns else np.nan
    return out


def _modal(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan


def collapse_to_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Class -> feature collapse: share/consistency = min across the 32 DFA
    classes (worst-case, matching the gate's universal-pass requirement);
    position = modal position across classes. Validated against the historical
    clean7/4061 split — reproduces it exactly (see workbook precedent)."""
    agg: dict[str, object] = {c: "min" for c in df.columns if c.startswith(("share_", "consistency_"))}
    for c in df.columns:
        if c.startswith("position_"):
            agg[c] = _modal
        elif c == "total_abs_R":
            agg[c] = "mean"
    return df.groupby("feature_idx", as_index=False).agg(agg)  # type: ignore[arg-type]


def merge_and_gate(dfa_feat: pd.DataFrame, z_feat: pd.DataFrame) -> pd.DataFrame:
    """Merge collapsed DFA/z feature tables and compute the membership gate:
    DFA share/consistency on R+C1+A, z share/consistency on R only (per Tom,
    15/07/26 — z C1/A columns aren't available for every config on disk)."""
    m = dfa_feat.merge(z_feat, on="feature_idx", suffixes=("_dfa", "_z"))
    ms, ef = GATE["min_share"], GATE["exact_frac"]
    m["is_member"] = (
        (m["share_R_dfa"] >= ms) & (m["share_C1_dfa"] >= ms) & (m["share_A_dfa"] >= ms) &
        (m["consistency_R_dfa"] == ef) & (m["consistency_C1_dfa"] == ef) & (m["consistency_A_dfa"] == ef) &
        (m["share_R_z"] >= ms) & (m["consistency_R_z"] == ef)
    )
    return m


def assert_position_agreement(m: pd.DataFrame, config_name: str) -> None:
    """Structural guard: a member passes the gate independently under DFA and
    z, at the same R-condition threshold — their locked positions should be
    forced to coincide. Fail loudly, don't assume, per project convention."""
    members = m[m["is_member"]]
    mismatched = members[members["position_R_dfa"] != members["position_R_z"]]
    assert mismatched.empty, (
        f"{config_name}: dfa_position != z_position for members "
        f"{mismatched['feature_idx'].tolist()} — investigate before trusting this config's output"
    )


def select_member_and_near_miss(m: pd.DataFrame) -> pd.DataFrame:
    """status column: 'member' for gate-passers, 'near_miss' for the top
    N_NEAR_MISS non-members by DFA share_R (universal, not just zero-pass
    configs — sort key is fixed regardless of per-config DFA/z divergence)."""
    members = m[m["is_member"]].copy()
    members["status"] = "member"
    rest = m[~m["is_member"]].sort_values("share_R_dfa", ascending=False)
    near_miss = rest.head(N_NEAR_MISS).copy()
    near_miss["status"] = "near_miss"
    return pd.concat([members, near_miss], ignore_index=True)


def class_mass_profiles(dfa_per_class: pd.DataFrame, feature_ids: list[int]) -> dict[int, tuple[list, list]]:
    """Top5/bottom5 classes by raw DFA mass_R (not share) — Tom's spec, shows
    which classes actually drive the feature. Computed only for the features
    that end up in the output (members + near-misses), not the whole dict."""
    profiles = {}
    sub = dfa_per_class[dfa_per_class["feature_idx"].isin(feature_ids)]
    for feat, g in sub.groupby("feature_idx"):
        ranked = g.sort_values("total_abs_R", ascending=False)
        top5 = list(zip(ranked["class_id"].head(N_CLASS_PROFILE).tolist(),
                         ranked["total_abs_R"].head(N_CLASS_PROFILE).round(6).tolist()))
        bottom5 = list(zip(ranked["class_id"].tail(N_CLASS_PROFILE).iloc[::-1].tolist(),
                            ranked["total_abs_R"].tail(N_CLASS_PROFILE).iloc[::-1].round(6).tolist()))
        profiles[int(feat)] = (top5, bottom5)  # type: ignore[arg-type]
    return profiles


OUTPUT_COLS = [
    "config", "feature_idx", "status",
    "dfa_share_R", "dfa_share_C1", "dfa_share_A",
    "dfa_consistency_R", "dfa_consistency_C1", "dfa_consistency_A", "dfa_position",
    "z_share_R", "z_share_C1", "z_share_A",
    "z_consistency_R", "z_consistency_C1", "z_consistency_A", "z_position",
    "top5_classes_dfa_r", "bottom5_classes_dfa_r",
]


def build_output_rows(subset: pd.DataFrame, profiles: dict, config_name: str) -> pd.DataFrame:
    """Rename merged/gated columns to the brief's exact per-feature output
    schema and attach the class mass profiles."""
    rows = []
    for _, r in subset.iterrows():
        top5, bottom5 = profiles[int(r["feature_idx"])]
        rows.append({
            "config": config_name, "feature_idx": int(r["feature_idx"]), "status": r["status"],
            "dfa_share_R": r["share_R_dfa"], "dfa_share_C1": r["share_C1_dfa"], "dfa_share_A": r["share_A_dfa"],
            "dfa_consistency_R": r["consistency_R_dfa"], "dfa_consistency_C1": r["consistency_C1_dfa"],
            "dfa_consistency_A": r["consistency_A_dfa"], "dfa_position": r["position_R_dfa"],
            "z_share_R": r["share_R_z"], "z_share_C1": r["share_C1_z"], "z_share_A": r["share_A_z"],
            "z_consistency_R": r["consistency_R_z"], "z_consistency_C1": r["consistency_C1_z"],
            "z_consistency_A": r["consistency_A_z"], "z_position": r["position_R_z"],
            "top5_classes_dfa_r": top5, "bottom5_classes_dfa_r": bottom5,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLS)


def _raw_feature_masses(mat: np.ndarray) -> np.ndarray:
    """Total |signed_vec_R| summed across clips, per feature — ranking metric
    only (selects the top-N ceiling pool), matches select_control_features.py's
    compute_feature_masses convention exactly."""
    return np.abs(mat).sum(axis=0)


def _combined_mass_pct_r(mat: np.ndarray, features: list[int]) -> float:
    """Self-consistent scaffold mass %, R condition only — same metric as
    scaffold_mass_pct.py (scaffold Sigma|signed_vec| / all-features Sigma|signed_vec|,
    per clip, mean across clips). Explicitly NOT the unresolved 36.5% metric."""
    total = np.abs(mat).sum(axis=1)
    scaffold_abs = np.abs(mat[:, features]).sum(axis=1)
    return float((scaffold_abs / total).mean())


def ceiling_check(cfg: dict, members: list[int]) -> dict:
    """Member configs only — ceiling = best possible R-mass any N-feature draw
    could achieve, using the actual top-N-by-mass features in the full dict.
    Reuses the 02/07 ceiling-collision method (select_control_features.py's
    ranking) with the self-consistent mass metric (scaffold_mass_pct.py's)."""
    if not members:
        return {"config": cfg["name"], "n_members": 0,
                "member_combined_mass_R": np.nan, "ceiling_combined_mass_R": np.nan, "pct_of_ceiling": np.nan}
    df = pd.read_parquet(_resolve_mass_delta_parquet(cfg))
    mat = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
    ranked = np.argsort(_raw_feature_masses(mat))[::-1]
    ceiling_features = ranked[:len(members)].tolist()
    member_mass = _combined_mass_pct_r(mat, members)
    ceiling_mass = _combined_mass_pct_r(mat, ceiling_features)
    return {"config": cfg["name"], "n_members": len(members),
            "member_combined_mass_R": member_mass, "ceiling_combined_mass_R": ceiling_mass,
            "pct_of_ceiling": member_mass / ceiling_mass * 100}


FULL_TABLE_COLS = [
    "feature_idx", "status",
    "share_R_dfa", "share_C1_dfa", "share_A_dfa",
    "consistency_R_dfa", "consistency_C1_dfa", "consistency_A_dfa", "position_R_dfa",
    "share_R_z", "share_C1_z", "share_A_z",
    "consistency_R_z", "consistency_C1_z", "consistency_A_z", "position_R_z",
]


def process_config(cfg: dict, out_dir: Path) -> dict:
    print(f"[{cfg['name']}] loading DFA/z scores…")
    dfa_per_class = load_dfa(cfg)
    z_per_class = load_z(cfg)
    merged = merge_and_gate(collapse_to_feature(dfa_per_class), collapse_to_feature(z_per_class))
    assert_position_agreement(merged, cfg["name"])

    full = merged.copy()
    full["status"] = np.where(full["is_member"], "member", "not_member")
    full[FULL_TABLE_COLS].to_csv(out_dir / f"{cfg['name']}_all_features.csv", index=False)

    subset = select_member_and_near_miss(merged)
    profiles = class_mass_profiles(dfa_per_class, subset["feature_idx"].astype(int).tolist())
    build_output_rows(subset, profiles, cfg["name"]).to_csv(out_dir / f"{cfg['name']}.csv", index=False)

    members = merged.loc[merged["is_member"], "feature_idx"].astype(int).tolist()
    print(f"[{cfg['name']}] {len(members)} members: {members}")
    # TF has no mass_delta_vm_c1 parquet (VM-only source) — ceiling_check's own
    # `if not members` short-circuit covers the expected TF=0 case; if TF ever
    # DID gate members, resolving the (VM-only) parquet path would raise, which
    # is correct — an unexpected TF pass is a finding to surface, not paper over.
    return ceiling_check(cfg, members)


def main() -> None:
    out_dir: Path = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ceiling_rows = [process_config(cfg, out_dir) for cfg in CONFIGS]

    ceiling_path = out_dir / "scaffold_selection_ceiling_summary.csv"
    pd.DataFrame(ceiling_rows, columns=[
        "config", "n_members", "member_combined_mass_R", "ceiling_combined_mass_R", "pct_of_ceiling",
    ]).to_csv(ceiling_path, index=False)
    print(f"\n→ {ceiling_path}")


if __name__ == "__main__":
    main()
