"""
feature_sign_balance.py

For a given feature, find classes where positive and negative DFA contributions
are well-balanced — useful for selecting visualisation targets where both
signs can be demonstrated within the same class.

Reports per class:
    n_clips_total   — all R-correct clips in this class
    n_active        — clips where |signed_vec_R[feat]| > 1e-8 (feature fires)
    frac_active     — n_active / n_clips_total
    mean_signed     — mean signed DFA contribution across active clips
    pct_positive    — % clips where signed_vec_R[feat] > 0
    pct_negative    — % clips where signed_vec_R[feat] < 0

Filter: n_active >= MIN_N, sorted by |mean_signed| ascending (most balanced first).

Usage:
    uv run python notebooks/feature_sign_balance.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
from ToT_utils import load_metadata

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE     = 5384
MIN_N       = 30          # minimum active clips to be shown
ACTIVE_EPS  = 1e-8

_pq_matches = list(ROOT.glob("outputs/analysis/**/dfa_mass_delta_vm_c1.parquet"))
assert len(_pq_matches) == 1, f"Expected 1 parquet, found: {_pq_matches}"
PARQUET = _pq_matches[0]
LABELS_PATH = ROOT / "data/ssv2/labels/labels.json"
VAL_PATH    = ROOT / "data/ssv2/labels/validation.json"
# ──────────────────────────────────────────────────────────────────────────────


def load_signed(feat: int) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET, columns=["clip_id", "class_id", "signed_vec_R"])
    df["signed"]    = df["signed_vec_R"].apply(lambda v: float(np.asarray(v)[feat]))
    df["abs_signed"] = df["signed"].abs()
    df["active"]    = df["abs_signed"] > ACTIVE_EPS
    return df


def class_names() -> dict[int, str]:
    label_map, _, _ = load_metadata(str(LABELS_PATH), str(VAL_PATH))
    return {v: k for k, v in label_map.items()}


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    total = df.groupby("class_id")["clip_id"].count().rename("n_clips_total")
    active_df = df[df["active"]]
    agg = active_df.groupby("class_id").agg(
        n_active    =("clip_id",    "count"),
        mean_signed =("signed",     "mean"),
        mean_abs    =("abs_signed", "mean"),
        pct_positive=("signed",     lambda x: (x > 0).mean()),
        pct_negative=("signed",     lambda x: (x < 0).mean()),
    )
    result = total.to_frame().join(agg, how="left").fillna(0).reset_index()
    result["frac_active"] = result["n_active"] / result["n_clips_total"]
    result["abs_mean"]    = result["mean_signed"].abs()
    return result


def main() -> None:
    names = class_names()
    df    = load_signed(FEATURE)
    result = summarise(df)
    result["class_name"] = result["class_id"].map(names)
    result = (result[result["n_active"] >= MIN_N]
              .sort_values("abs_mean")
              .reset_index(drop=True))

    cols = ["class_id", "class_name", "n_clips_total", "n_active",
            "frac_active", "mean_abs", "mean_signed", "pct_positive", "pct_negative"]
    print(f"Feature {FEATURE} — classes with n_active ≥ {MIN_N}, sorted by |mean_signed|")
    print()
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_colwidth", 50)
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
