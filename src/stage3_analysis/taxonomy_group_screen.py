"""
Track B — taxonomy group screen. Cheap, existing-data-only candidate screen:
scores every feature on (1) how concentrated its DFA mass is across the
32-class SL set (taxonomy-free: entropy, participation ratio), and (2) whether
that mass concentrates inside one of six Goyal et al. 2017 (Table 8,
arXiv:1706.04261) action-groups (taxonomy-gated: in/out-group mass ratio).

R condition only — screen, not perturbation analysis. No new extraction; pure
pandas over dfa_mass_delta_vm_c1.parquet (VM L7 job64) and dfa_mass_delta.parquet
(TF L7, its only config). L7 x8k64 both backbones only, for direct comparability.

Two anchors validated on stdout (not asserted — this is a screen, not a locked
test): clean7 expected flat (high entropy/PR, no group elevation), all4 expected
structured (low entropy/PR, elevated on Covering — partial pass, see brief).

Outputs (outputs/analysis/taxonomy_group_screen/):
    track_b_feature_specificity.csv   — entropy/PR/total_mass_R, one row per (backbone, feature)
    track_b_group_ratio.csv           — in/out-group mass ratio, one row per (backbone, group, feature)

Usage:
    uv run python src/stage3_analysis/taxonomy_group_screen.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

DFA_CLASSES = sorted({0, 6, 14, 18, 19, 23, 27, 28, 29, 30, 31, 32, 36, 37, 40,
                       41, 42, 44, 57, 59, 83, 84, 123, 126, 142, 143, 145, 164,
                       168, 169, 171, 173})

TAXONOMY_GROUPS = {
    "camera_motions":       [0, 32, 41, 44, 168],
    "lifting_and_dropping": [27, 28, 30, 31],
    "moving_two_objects":   [36, 37, 40, 42],
    "covering":             [6, 171],
    "holding":              [18, 19],
    "turning_upside_down":  [84, 164],
}

CLEAN7 = [1842, 5578, 1996, 3513, 1990, 3558, 5552]
ALL4   = [6135, 2197, 1535, 308]

CFG = {
    "vm_dir":  ROOT / "outputs/analysis/dfa_mass_delta_vm_c1",
    "tf_path": ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
    "out_dir": ROOT / "outputs/analysis/taxonomy_group_screen",
}


def _resolve_source(backbone: str) -> Path:
    """VM: suffixed L7/job64/k64 pattern preferred, legacy unsuffixed fallback
    (matches every other script's convention). TF: only one L7 config exists —
    dfa_mass_delta.py hardcodes it, so no suffix ambiguity to resolve."""
    if backbone == "tf":
        path = CFG["tf_path"]
    else:
        suffixed = CFG["vm_dir"] / "dfa_mass_delta_vm_c1_l7_job64_k64.parquet"
        legacy   = CFG["vm_dir"] / "dfa_mass_delta_vm_c1.parquet"
        path = suffixed if suffixed.exists() else legacy
    if not path.exists():
        raise FileNotFoundError(f"No source parquet for backbone={backbone}: {path}")
    return path


def mass_matrix(path: Path) -> np.ndarray:
    """mean_abs_R per (class_id, feature_idx), R condition only — matches the
    existing convention (per_class_feature_delta.py's compute_feature_stats).
    Returns (32, dict_size), rows ordered by DFA_CLASSES (sorted, fixed order)."""
    df = pd.read_parquet(path, columns=["class_id", "signed_vec_R"])
    df = df[df["class_id"].isin(DFA_CLASSES)]
    rows = []
    for cls in DFA_CLASSES:
        mat = np.stack(df.loc[df["class_id"] == cls, "signed_vec_R"].to_numpy()).astype(np.float32)
        rows.append(np.abs(mat).mean(axis=0))
    return np.stack(rows)   # (32, dict_size)


def compute_specificity(mass_mat: np.ndarray, backbone: str) -> pd.DataFrame:
    """Axis 1, taxonomy-free — entropy + participation ratio, vectorised over
    all features at once (mass_mat is (32 classes, dict_size))."""
    n_classes  = mass_mat.shape[0]
    total_mass = mass_mat.sum(axis=0)                       # (dict_size,)
    p = mass_mat / np.clip(total_mass, 1e-8, None)           # (32, dict_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=0)
    normalized_entropy = entropy / np.log(n_classes)

    sq_sum = (mass_mat ** 2).sum(axis=0)
    participation_ratio = total_mass ** 2 / np.clip(sq_sum, 1e-8, None)
    normalized_pr = participation_ratio / n_classes

    return pd.DataFrame({
        "backbone": backbone, "feature_idx": np.arange(mass_mat.shape[1]),
        "entropy": entropy, "normalized_entropy": normalized_entropy,
        "participation_ratio": participation_ratio, "normalized_pr": normalized_pr,
        "total_mass_R": total_mass,
    })


def compute_group_ratio(mass_mat: np.ndarray, backbone: str) -> pd.DataFrame:
    """Axis 2, taxonomy-gated — in/out-group mass ratio per (feature, group).
    Long format: six groups is small enough that wide columns would just be
    sparse for the 19 taxonomy-singleton classes not scored against any group."""
    class_row = {c: i for i, c in enumerate(DFA_CLASSES)}
    chunks = []
    for group, classes in TAXONOMY_GROUPS.items():
        in_idx  = [class_row[c] for c in classes]
        out_idx = [i for c, i in class_row.items() if c not in classes]
        in_mass  = mass_mat[in_idx, :].mean(axis=0)
        out_mass = mass_mat[out_idx, :].mean(axis=0)
        ratio = in_mass / (out_mass + 1e-8)
        chunk = pd.DataFrame({
            "backbone": backbone, "group": group, "feature_idx": np.arange(mass_mat.shape[1]),
            "in_group_mass": in_mass, "out_group_mass": out_mass, "ratio": ratio,
        })
        chunk["rank_in_group"] = chunk["ratio"].rank(ascending=False, method="min").astype(int)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def print_validation(specificity: pd.DataFrame, group_ratio: pd.DataFrame) -> None:
    """clean7 (expect flat) vs all4 (expect structured, elevated on Covering)
    — VM L7 job64 only. Eyeball check, not asserted (this is a screen)."""
    vm_spec = specificity[specificity["backbone"] == "vm"]
    vm_grp  = group_ratio[group_ratio["backbone"] == "vm"]
    wide_ratio = vm_grp.pivot(index="feature_idx", columns="group", values="ratio")

    for label, features in [("clean7 (expect FLAT)", CLEAN7), ("all4 (expect STRUCTURED, Covering-elevated)", ALL4)]:
        print(f"\n=== {label} ===")
        sub = vm_spec[vm_spec["feature_idx"].isin(features)].set_index("feature_idx")
        view = sub[["entropy", "normalized_entropy", "participation_ratio", "normalized_pr"]] \
            .join(wide_ratio.loc[features])
        print(view.round(4).to_string())


def main() -> None:
    out_dir: Path = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    specificity_frames, group_frames = [], []
    for backbone in ["vm", "tf"]:
        print(f"[{backbone}] loading mass matrix…")
        mat = mass_matrix(_resolve_source(backbone))
        specificity_frames.append(compute_specificity(mat, backbone))
        group_frames.append(compute_group_ratio(mat, backbone))

    specificity = pd.concat(specificity_frames, ignore_index=True)
    group_ratio = pd.concat(group_frames, ignore_index=True) \
        .sort_values(["backbone", "group", "rank_in_group"]).reset_index(drop=True)

    specificity.to_csv(out_dir / "track_b_feature_specificity.csv", index=False)
    group_ratio.to_csv(out_dir / "track_b_group_ratio.csv", index=False)
    print(f"\n{len(specificity):,} rows -> track_b_feature_specificity.csv")
    print(f"{len(group_ratio):,} rows -> track_b_group_ratio.csv")

    print_validation(specificity, group_ratio)


if __name__ == "__main__":
    main()
