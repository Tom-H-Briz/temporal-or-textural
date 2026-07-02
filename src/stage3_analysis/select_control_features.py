"""
Control feature selection for scaffold ablation study.

Rejection-samples an 8-feature mass-matched control set from non-scaffold
features. Target = combined total |signed_vec_R| of ALL8 across 3,558 clips.
Eligible pool = top-N non-scaffold features by the same metric (mass-ranked
so the draw can realistically hit the high-mass scaffold target).

Output: outputs/analysis/scaffold_ablation/control_feature_selection.json

Usage:
    uv run python src/stage3_analysis/select_control_features.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent

ALL8 = [1842, 5578, 1996, 3513, 1990, 3558, 4061, 5552]

CFG = {
    "source_glob":    "outputs/analysis/**/dfa_mass_delta_vm_c1.parquet",
    "out_dir":        ROOT / "outputs/analysis/scaffold_ablation",
    "eligible_pool_n": 300,
    "tolerance":      0.10,
    "seed":           42,
    "max_attempts":   10_000,
    "draw_size":      8,
}


def compute_feature_masses(df: pd.DataFrame) -> np.ndarray:
    """Total |signed_vec_R| summed across all clips. Returns (6144,) float32."""
    mat = np.stack(df["signed_vec_R"].to_numpy()).astype(np.float32)
    return np.abs(mat).sum(axis=0)


def select_control(masses: np.ndarray, cfg: dict) -> dict:
    scaffold_set = set(ALL8)
    target_mass  = float(masses[ALL8].sum())

    non_scaffold = np.array([i for i in range(len(masses)) if i not in scaffold_set])
    ranked       = non_scaffold[np.argsort(masses[non_scaffold])[::-1]]
    n_eligible   = min(cfg["eligible_pool_n"], len(ranked))
    eligible     = ranked[:n_eligible]

    rng = np.random.default_rng(cfg["seed"])
    for attempt in range(1, cfg["max_attempts"] + 1):
        draw     = rng.choice(eligible, size=cfg["draw_size"], replace=False)
        achieved = float(masses[draw].sum())
        if abs(achieved - target_mass) / target_mass <= cfg["tolerance"]:
            return {
                "target_mass":       target_mass,
                "tolerance":         cfg["tolerance"],
                "seed":              cfg["seed"],
                "eligible_pool_size": n_eligible,
                "n_attempts":        attempt,
                "selected_features": sorted(int(f) for f in draw),
                "achieved_mass":     achieved,
                "pct_deviation":     float((achieved - target_mass) / target_mass * 100),
            }

    raise RuntimeError(
        f"No match found in {cfg['max_attempts']:,} attempts — "
        f"widen eligible_pool_n (currently {cfg['eligible_pool_n']}) or tolerance."
    )


def main() -> None:
    out_dir: Path = CFG["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    matches = list(ROOT.glob(CFG["source_glob"]))
    assert len(matches) == 1, f"Expected 1 source parquet, found: {matches}"
    print(f"Source: {matches[0]}")

    df     = pd.read_parquet(matches[0])
    masses = compute_feature_masses(df)
    print(f"  {len(df):,} clips  |  ALL8 target mass: {masses[ALL8].sum():.4f}")

    result = select_control(masses, CFG)
    print(f"  Selected in {result['n_attempts']:,} attempts: {result['selected_features']}")
    print(f"  Achieved: {result['achieved_mass']:.4f}  ({result['pct_deviation']:+.2f}% from target)")

    out_path = out_dir / "control_feature_selection.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
