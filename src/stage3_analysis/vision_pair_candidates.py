"""Candidate selection (CPU) after umt_vision_pair_aggregate.py, per pair (§7b recipe).

From feature_table.csv + clip_feature_arrays_R.npz (+ optional ramp_check.csv):
  dfa_share   mean |dfa| per feature / mean total |dfa| per clip (R condition)
  screen      keep the top `top_frac` of features by dfa_share (default 5%)
  rank        evidence (helps true class), |auroc - 0.5| (level signature), |bias|
  flag        anti-evidence: negative evidence and helps_true_rate < 0.5
Writes candidates_dfa.csv (screened table, dfa_share + ramp columns joined) and prints
shortlists to pick figure ids from. Usage:
  PAIR=42,40 uv run python src/stage3_analysis/vision_pair_candidates.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_M = os.environ.get("MODEL", "umt")
_PAIR = tuple(int(x) for x in os.environ.get("PAIR", "45,43").split(","))
_SUF = "" if _PAIR == (45, 43) else f"_{_PAIR[0]}_{_PAIR[1]}"
CFG = {
    "in_dir":   ROOT / "outputs/analysis" / (("umt_vision_pair_contrast" if _M == "umt" else f"{_M}_vision_pair_contrast") + _SUF),
    "top_frac": 0.05,   # keep top 5% by dfa_share, then rank (§7b)
    "n_print":  15,
}


def dfa_shares(in_dir):
    """dfa_share per feature from the R-condition clip arrays (mean |dfa| / mean clip total |dfa|)."""
    A = np.load(in_dir / "clip_feature_arrays_R.npz")
    dfa = A["dfa"]
    return pd.Series(np.abs(dfa).mean(0) / np.abs(dfa).sum(1).mean(), name="dfa_share")


def screen(ft, shares, cfg):
    """Join dfa_share, keep the top `top_frac` by it, add ranking + anti-evidence columns."""
    t = ft.drop(columns=["dfa_share", "level_sig"], errors="ignore").merge(
        shares, left_on="feature", right_index=True)
    cut = t.dfa_share.quantile(1 - cfg["top_frac"])
    t = t[t.dfa_share >= cut].copy()
    t["level_sig"] = (t.auroc - 0.5).abs()
    t["anti"] = (t.evidence < 0) & (t.helps_true_rate < 0.5)
    ramp = cfg["in_dir"] / "ramp_check.csv"
    if ramp.exists():                       # gate / ramp_shuffle columns if the ramp check ran
        t = t.merge(pd.read_csv(ramp), on="feature", how="left", suffixes=("", "_ramp"))
    return t.sort_values("dfa_share", ascending=False)


def show(t, cfg):
    """Print the shortlists the §7b recipe ranks by: evidence, level signature, bias, anti."""
    cols = ["feature", "dfa_share", "evidence", "auroc", "bias", "helps_true_rate",
            "rev_swap", "c_keep"] + (["gate", "ramp_shuffle"] if "gate" in t else [])
    n = cfg["n_print"]
    for name, tab in (("evidence", t.sort_values("evidence", ascending=False).head(n)),
                      ("level (|auroc-.5|)", t.sort_values("level_sig", ascending=False).head(n)),
                      ("bias", t.reindex(t.bias.abs().sort_values(ascending=False).index).head(n)),
                      ("anti-evidence", t[t.anti].sort_values("evidence").head(10))):
        print(f"\n== top by {name} ==")
        print(tab[cols].round(3).to_string(index=False))


def main():
    t = screen(pd.read_csv(CFG["in_dir"] / "feature_table.csv"), dfa_shares(CFG["in_dir"]), CFG)
    t.to_csv(CFG["in_dir"] / "candidates_dfa.csv", index=False)
    print(f"{len(t)} features kept (top {CFG['top_frac']:.0%} by dfa_share)")
    pd.set_option("display.width", 250)
    show(t, CFG)


if __name__ == "__main__":
    main()
