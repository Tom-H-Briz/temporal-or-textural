"""
Mass concentration ratio — empirical random-draw control for scaffold DFA mass.

A theoretical n/dict_size baseline only gives the *mean* of a random N-feature
draw's mass share; it says nothing about whether a heavy-tailed mass
distribution could let an unlucky/lucky draw land near the scaffold's actual
mass by chance. This draws real random N-feature subsets from each config's
real per-feature mass distribution (not an assumed uniform one) and reports
where the scaffold sits in that empirical distribution.

Reuses scaffold_selection_consolidated.py's exact mass metric and mass-delta
file resolver, and reads live membership from its {config}.csv outputs
(status == "member"), so this can't drift out of sync with the gate.

Outputs:
    outputs/analysis/mass_concentration_bootstrap/mass_concentration_bootstrap.csv

Usage:
    uv run python src/stage3_analysis/mass_concentration_bootstrap.py
"""

import sys
import zlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "stage3_analysis"))
from scaffold_selection_consolidated import (
    CONFIGS, _resolve_mass_delta_parquet, _combined_mass_pct_r,
)

CFG = {
    "seed":            0,
    "n_draws":         2000,
    "selection_dir":   ROOT / "outputs/analysis/scaffold_selection",
    "out_dir":         ROOT / "outputs/analysis/mass_concentration_bootstrap",
}


def load_members(config_name: str) -> list[int]:
    """Live membership for a config, read from scaffold_selection_consolidated.py's
    own {config}.csv output — never hardcoded, so this can't go stale relative
    to the gate."""
    path = CFG["selection_dir"] / f"{config_name}.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.loc[df["status"] == "member", "feature_idx"].tolist()


def _rng_for(config_name: str) -> np.random.Generator:
    """Per-config RNG keyed by (seed, config name) via a stable CRC32 salt —
    not Python's hash(), which is randomized per-process by default. Draws for
    one config are therefore independent of every other config's draws and of
    CONFIGS' ordering, while staying reproducible run to run."""
    salt = zlib.crc32(config_name.encode())
    return np.random.default_rng([CFG["seed"], salt])


def bootstrap_config(cfg: dict, members: list[int]) -> dict:
    """Empirical mass-concentration control for one config: member mass vs.
    N_DRAWS real random N-feature draws from the actual (skewed) per-feature
    mass distribution, not a uniform-mean assumption."""
    mat = np.stack(
        pd.read_parquet(_resolve_mass_delta_parquet(cfg))["signed_vec_R"].to_numpy()
    ).astype(np.float32)
    dict_size = mat.shape[1]
    member_mass = _combined_mass_pct_r(mat, members)
    rng = _rng_for(cfg["name"])
    draws = np.array([
        _combined_mass_pct_r(mat, rng.choice(dict_size, size=len(members), replace=False).tolist())
        for _ in range(CFG["n_draws"])
    ])
    return {"config": cfg["name"], "n_members": len(members), "dict_size": dict_size,
            "member_mass_pct": member_mass * 100, "boot_mean_pct": draws.mean() * 100,
            "boot_std_pct": draws.std() * 100, "boot_max_pct": draws.max() * 100,
            "z_score": (member_mass - draws.mean()) / draws.std(),
            "pct_random_draws_beating_scaffold": (draws >= member_mass).mean() * 100}


def main():
    rows = []
    for cfg in CONFIGS:
        members = load_members(cfg["name"])
        if not members:
            continue
        try:
            rows.append(bootstrap_config(cfg, members))
        except FileNotFoundError:
            continue
    out = pd.DataFrame(rows)
    CFG["out_dir"].mkdir(parents=True, exist_ok=True)
    out.to_csv(CFG["out_dir"] / "mass_concentration_bootstrap.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
