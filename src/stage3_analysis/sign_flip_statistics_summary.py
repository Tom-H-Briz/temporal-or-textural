"""
Consolidated sign-flip statistics — Mann-Whitney dose-response and bucket-
fraction logistic regression, both backbones, every version, one file.

Two separately-computed analyses answering the same underlying question (does
sign-flip status specifically predict behavioural failure under shuffle, not
just any active top-10 feature) from different angles:
  - Mann-Whitney: does sign-flip MAGNITUDE (swing) differ between clips that
    fail vs. survive shuffle? (signflip_dose_response.py's own output)
  - Logistic regression: does sign-flip PRESENCE (frac_sign_flip) predict
    P(correct|shuffle) more than decrease/increase presence, holding all
    three in one model, pooled and with class fixed effects?
    (clip_shuffle_disruption.py's own output)

Ablation-based contrast set (a matched-frequency "stable" control, e.g.
STABLE12) abandoned (Tom, 24/08) — no clean construction exists: the top 24
TF features by top-10 frequency are ALL flip-capable, and the first
genuinely zero-flip feature is 5-45x less frequent than TOP12, so no set is
simultaneously frequency-matched and truly non-flipping. The two
correlational analyses below are the finalized evidence for this claim.

Reads only already-computed CSVs — no new extraction, no new regression.

Outputs:
    outputs/analysis/shuffle_reduction_composition/sign_flip_statistics_summary.csv

Usage:
    uv run python src/stage3_analysis/sign_flip_statistics_summary.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"

CFG = {
    "dose_response": {"vm": DIR / "vm_dose_response_summary.csv",
                       "tf": DIR / "tf_dose_response_summary.csv"},
    "bucket_logit":  {"vm": DIR / "ssv2_vm_logit_coefficients.csv",
                       "tf": DIR / "ssv2_tf_logit_coefficients.csv"},
    "out_path": DIR / "sign_flip_statistics_summary.csv",
}


def load_dose_response(backbone: str) -> pd.DataFrame:
    """n_failing/n_surviving are the test's own instance-count columns, kept
    as-is; "n" here is just their sum, added so every row in the combined
    summary has a comparable total-sample-size column, not NaN for this test."""
    df = pd.read_csv(CFG["dose_response"][backbone])
    df.insert(0, "backbone", backbone.upper())
    df.insert(1, "dataset", "ssv2")
    df.insert(2, "test", "mann_whitney_dose_response")
    df.insert(3, "term", "swing")  # the one quantity this test compares (failing vs surviving)
    df.insert(5, "n", df["n_failing"] + df["n_surviving"])
    return df


def add_reference_category_rows(df: pd.DataFrame) -> pd.DataFrame:
    """frac_noise + frac_sign_flip + frac_decrease + frac_increase = 1 for every
    clip (full partition of its top-10 mass) — including all 4 as regressors
    alongside an intercept would be perfectly collinear, so one is always
    dropped as reference. That's frac_noise here; its "coefficient" is fixed
    at 0 by construction (const already captures its baseline effect), added
    explicitly so all 4 categories are visible rather than one silently
    missing from the table."""
    ref_rows = df[df["term"] == "const"].copy()
    ref_rows["term"] = "frac_noise"
    for col in ("coef", "se", "z"):
        ref_rows[col] = 0.0
    ref_rows["p"] = pd.NA
    ref_rows["notes"] = "reference category (fixed at 0 by construction; const captures its baseline)"
    return pd.concat([df, ref_rows], ignore_index=True)


def load_bucket_logit(backbone: str) -> pd.DataFrame:
    df = pd.read_csv(CFG["bucket_logit"][backbone])
    df.insert(0, "backbone", backbone.upper())
    df.insert(1, "dataset", "ssv2")
    df.insert(2, "test", "bucket_fraction_logit")
    df = df.drop(columns=["config"])
    return add_reference_category_rows(df)


def main() -> None:
    frames = [load_dose_response(bb) for bb in ("vm", "tf")]
    frames += [load_bucket_logit(bb) for bb in ("vm", "tf")]
    out = pd.concat(frames, ignore_index=True)

    col_order = ["backbone", "dataset", "test", "version", "term", "n", "p"]
    out = out[col_order + [c for c in out.columns if c not in col_order]]
    out.to_csv(CFG["out_path"], index=False)

    pd.set_option("display.width", 200)
    print(out[["backbone", "test", "version", "term", "n", "p"]].to_string(index=False))
    print(f"\n-> {CFG['out_path']}")


if __name__ == "__main__":
    main()
