"""
Scaffold sign-consistency per class — VM, any layer/sae_k with a scaffold
membership table (outputs/analysis/scaffold_selection/L{layer}_x{expansion}
k{sae_k}_VM.csv).

Source: dfa_mass_delta_vm_c1's per-clip signed_vec_R — the only per-clip
signed-DFA data persisted to disk (position_lock_extraction.py sums per-clip
signed DFA into running per-class totals and discards the per-clip values).
signed_vec_R sums DFA over ALL tokens in the clip, not just the locked
position — used as a proxy since scaffold members gate on share_R >= 0.90
and frac_clips_matching_mode_R == 1.0, so the locked position dominates the
whole-clip sum for the large majority of clips.

Usage:
    uv run python src/stage3_analysis/scaffold_sign_consistency.py --layer 5 --sae-k 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))
from stage3_analysis.scaffold_selection_consolidated import _resolve_mass_delta_parquet
from ToT_utils import _SAE_EXPANSION_FOR_K

CFG = {
    "dataset": "ssv2",
    "active_threshold": 1e-8,  # matches per_class_feature_delta.py's inactive-clip exclusion
    "selection_dir": ROOT / "outputs/analysis/scaffold_selection",
    "out_dir": ROOT / "outputs/analysis/scaffold_sign_consistency",
}


def load_locked_positions(cfg: dict, layer: int, sae_k: int) -> dict[int, int]:
    """feature_id -> locked position, from scaffold_selection_consolidated.py's
    membership table for this layer/sae_k — status=="member" rows only."""
    expansion = _SAE_EXPANSION_FOR_K[sae_k]
    path = cfg["selection_dir"] / f"L{layer}_x{expansion}k{sae_k}_VM.csv"
    df = pd.read_csv(path)
    members = df[df["status"] == "member"]
    return dict(zip(members["feature_idx"], members["dfa_position"]))


def resolve_mass_delta_path(layer: int, sae_k: int, dataset: str) -> Path:
    """Reuses scaffold_selection_consolidated.py's resolver (same lookup order,
    can't drift out of sync) — but warns loudly if it falls back to a pre-fix
    legacy file, since that's exactly the stale-pairing bug that resolver's
    own docstring warns never to prefer."""
    name = f"l{layer}_k{sae_k}"
    path = _resolve_mass_delta_parquet({"layer": layer, "sae_k": sae_k, "dataset": dataset, "name": name})
    if "job7ep" not in path.name:
        print(f"  WARNING: no post-fix mass-delta parquet for l{layer}_k{sae_k} — "
              f"falling back to pre-fix legacy file {path.name}")
    return path


def class_feature_sign_stats(clips: pd.DataFrame, feature_id: int, position: int, threshold: float) -> dict:
    """Modal sign / consistency / flip count for one (class, feature) pair.
    Inactive clips (|signed_vec_R[f]| <= threshold) don't vote either way —
    same exclusion per_class_feature_delta.py's sign_R uses."""
    vecs   = np.stack(clips["signed_vec_R"].values)[:, feature_id]
    active = vecs[np.abs(vecs) > threshold]
    n_pos, n_clips = int((active > 0).sum()), len(active)
    modal_sign = 1 if n_pos >= n_clips - n_pos else -1
    matching   = n_pos if modal_sign == 1 else n_clips - n_pos
    return {
        "feature_id": feature_id, "locked_position": position, "n_clips": n_clips,
        "modal_sign": modal_sign,
        "consistency": matching / n_clips if n_clips else float("nan"),
        "flip_count": n_clips - matching,
    }


def build_table(df: pd.DataFrame, locked_position: dict[int, int], threshold: float) -> pd.DataFrame:
    rows = []
    for class_id, class_clips in df.groupby("class_id"):
        sl_label = class_clips["sl_label"].iloc[0]
        for feature_id, position in locked_position.items():
            stats = class_feature_sign_stats(class_clips, feature_id, position, threshold)
            rows.append({"class_id": class_id, "sl_label": sl_label, **stats})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sae-k", type=int, default=64)
    args = parser.parse_args()
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    locked_position = load_locked_positions(cfg, args.layer, args.sae_k)
    if not locked_position:
        print(f"  No scaffold members for l{args.layer}_k{args.sae_k} — nothing to compute")
        return

    parquet = resolve_mass_delta_path(args.layer, args.sae_k, cfg["dataset"])
    df = pd.read_parquet(parquet, columns=["class_id", "sl_label", "signed_vec_R"])
    table = build_table(df, locked_position, cfg["active_threshold"])
    out_path = cfg["out_dir"] / f"sign_consistency_l{args.layer}_vm_k{args.sae_k}.csv"
    table.to_csv(out_path, index=False)
    print(f"→ {out_path}  ({len(table)} rows, {df['class_id'].nunique()} classes, "
          f"{len(locked_position)} features)")


if __name__ == "__main__":
    main()
