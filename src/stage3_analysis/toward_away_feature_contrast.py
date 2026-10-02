"""
Key SAE features for the toward/away mirror pair (classes 42 vs 40).

"Moving something closer to something" vs "Moving something away from
something" — the pair where position-embedding reversal FLIPS predictions
(45-51% of baseline-correct clips). This identifies the DFA features that
carry that direction signal: per-class mean signed DFA (R condition), the
opposite-sign contrast across the pair, temporal survival under C1 (frame-pair
shuffle) and A (single frame), and each top feature's 8-tubelet DFA profile
(Spearman ramp correlation — monotonic ramp ties it to the reversal result).

Inputs: dfa_mass_delta_vm_c1_l5_job7ep_k64.parquet (per-clip signed_vec R/C1/A,
4,450 R-correct clips) + position_lock_videomae_ssv2_l5_job7ep_k64.parquet
(per class x feature x tubelet). Scaffold/chiral flags for context.

Usage: uv run python src/stage3_analysis/toward_away_feature_contrast.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.ablation_targets import get_targets

CFG = {
    "dfa_parquet": ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/"
                          "dfa_mass_delta_vm_c1_l5_job7ep_k64.parquet",
    "pos_parquet": ROOT / "outputs/analysis/position_lock/"
                          "position_lock_videomae_ssv2_l5_job7ep_k64.parquet",
    "chiral_csv": ROOT / "outputs/analysis/chiral/chiral_features_contrast.csv",
    "closer": 42, "away": 40,
    "top_class": 10, "top_contrast": 20,
    "out_csv": ROOT / "outputs/analysis/explore/toward_away_feature_contrast_l5.csv",
}


def class_mean_vec(df: pd.DataFrame, class_id: int, col: str) -> np.ndarray:
    """Mean signed-DFA vector (6,144) for one class and condition."""
    v = np.vstack(df[df.class_id == class_id][col].values)
    return v.mean(axis=0)


def build_feature_table(cfg: dict) -> pd.DataFrame:
    """One row per feature: mean signed DFA (closer/away) under R, C1, A, plus
    opposite-sign contrast scores in both directions."""
    df = pd.read_parquet(cfg["dfa_parquet"],
                         columns=["class_id", "signed_vec_R",
                                  "signed_vec_C1", "signed_vec_A"])
    vecs = {(cid, cond): class_mean_vec(df, cid, f"signed_vec_{cond}")
            for cid in (cfg["closer"], cfg["away"]) for cond in ("R", "C1", "A")}
    t = pd.DataFrame({"feature_idx": np.arange(len(vecs[(cfg["closer"], "R")]))})
    for cid, tag in [(cfg["closer"], "closer"), (cfg["away"], "away")]:
        for cond in ("R", "C1", "A"):
            t[f"mean_{cond}_{tag}"] = vecs[(cid, cond)]
    t["toward_score"] = np.minimum(t.mean_R_closer, -t.mean_R_away)
    t["away_score"] = np.minimum(-t.mean_R_closer, t.mean_R_away)
    return t


def add_profiles_and_flags(t: pd.DataFrame, cfg: dict, feats: list[int]) -> pd.DataFrame:
    """Spearman(tubelet idx, mean signed DFA per tubelet) per class for the
    selected features, plus scaffold / chiral-list membership flags."""
    pos = pd.read_parquet(cfg["pos_parquet"],
                          columns=["class_id", "feature_idx", "tubelet_idx",
                                   "mean_dfa_signed_R"])
    pos = pos[pos.feature_idx.isin(feats)]
    sub = t[t.feature_idx.isin(feats)].copy()
    for cid, tag in [(cfg["closer"], "closer"), (cfg["away"], "away")]:
        piv = pos[pos.class_id == cid].pivot(
            index="feature_idx", columns="tubelet_idx", values="mean_dfa_signed_R")
        sub[f"ramp_rho_{tag}"] = [stats.spearmanr(range(8), piv.loc[f]).statistic
                                  if f in piv.index else np.nan for f in sub.feature_idx]
    scaffold = get_targets("ssv2", 5)["all7"]
    sub["in_l5_scaffold"] = sub.feature_idx.isin(scaffold)
    if cfg["chiral_csv"].exists():
        chiral = pd.read_csv(cfg["chiral_csv"])
        col = "feature_idx" if "feature_idx" in chiral.columns else chiral.columns[0]
        sub["in_chiral_lr"] = sub.feature_idx.isin(chiral[col])
    return sub


def main() -> None:
    cfg = CFG
    t = build_feature_table(cfg)
    picks = set()
    for col, k in [("mean_R_closer", cfg["top_class"]), ("mean_R_away", cfg["top_class"]),
                   ("toward_score", cfg["top_contrast"]), ("away_score", cfg["top_contrast"])]:
        picks |= set(t.nlargest(k, col).feature_idx)
    out = add_profiles_and_flags(t, cfg, sorted(picks))
    cfg["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["out_csv"], index=False)

    show = ["feature_idx", "mean_R_closer", "mean_R_away", "mean_C1_closer",
            "mean_A_closer", "ramp_rho_closer", "ramp_rho_away"]
    for title, col in [("TOP FEATURES — closer (42)", "mean_R_closer"),
                       ("TOP FEATURES — away (40)", "mean_R_away"),
                       ("TOWARD DETECTORS (+closer, -away)", "toward_score"),
                       ("AWAY DETECTORS (-closer, +away)", "away_score")]:
        print(f"\n{title}")
        cols = show if col.startswith("mean") else show + [col]
        print(t.nlargest(10, col)[["feature_idx", col] +
              [c for c in cols if c != col and c in t.columns]].to_string(index=False))
    print(f"\n{len(out)} features -> {cfg['out_csv']}")


if __name__ == "__main__":
    main()
