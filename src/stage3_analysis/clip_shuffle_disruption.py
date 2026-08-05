"""
Per-clip, sign-aware shuffle disruption vs behavioural loss — CC brief 05/08 addendum.

Step 1 (empirical, run first): checks whether clip-level DFA reliability regime
(max_mean_ratio) predicts collapsed top-10 magnitudes — the failure mode that
broke the clean7 mean-based headline (06/07). Finding: it doesn't, in either
backbone — top-10-by-magnitude selection already protects against near-zero
denominators. Relative-only 5% threshold used, no absolute floor.

Step 2: four-way per-clip classification (noise -> sign_flip -> decrease/increase,
in that order) of each R-correct clip's own top-10 |signed_R| features.

Step 3: per-backbone logistic regression, P(correct_under_shuffle) ~ frac_sign_flip
+ frac_decrease + frac_increase, frac_noise as reference. VM-SSv2 and TF-SSv2 only
— K400 excluded this round (see brief).

Usage:
    uv run python src/stage3_analysis/clip_shuffle_disruption.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).parent.parent.parent

ELIGIBILITY_THRESHOLD = 0.40
TOP_N = 10
CHANGE_BAND = 0.05  # relative-only, no absolute floor — see Step 1 finding above
OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"

CONFIGS = {
    "ssv2_vm": dict(
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_l7_job7ep_k64.parquet",
        shuffle_col="signed_vec_C1",
        correct_col="correct_C1",
        r_acc_csv=ROOT / "outputs/stage1_class_selection_VM_ssv2/per_class_accuracy_VM_ssv2_R.csv",
        reliability_parquet=None,  # no checkpoint-consistent diagnostic exists — computed inline instead
    ),
    "ssv2_tf": dict(
        dfa_parquet=ROOT / "outputs/analysis/dfa_mass_delta/dfa_mass_delta.parquet",
        shuffle_col="signed_vec_C",
        correct_col="correct_C",
        r_acc_csv=ROOT / "outputs/stage1_class_selection_TF/per_class_accuracy_TF.csv",
        reliability_parquet=ROOT / "outputs/analysis/cumulative_mass_diagnostic_tf_l7_k64_x8.parquet",
    ),
}


def eligible_classes(dfa_df: pd.DataFrame, r_acc_csv: Path) -> list[int]:
    r_acc_df = pd.read_csv(r_acc_csv)
    passed = set(r_acc_df.loc[r_acc_df["accuracy"] >= ELIGIBILITY_THRESHOLD, "class_id"])
    present = set(dfa_df["class_id"].unique())
    return sorted(passed & present)


def max_mean_ratio(signed_r_col: pd.Series) -> np.ndarray:
    """Per-clip max(|signed_R|) / mean(|signed_R|) over all 6144 dims — no active-token
    filter (unlike the original diagnostic) since token_fire_counts isn't stored in
    these parquets; used only for VM, which has no checkpoint-consistent diagnostic on disk."""
    return np.array([np.abs(v).max() / np.abs(v).mean() for v in signed_r_col])


def top10_floor(signed_r_col: pd.Series) -> np.ndarray:
    return np.array([np.sort(np.abs(v))[::-1][:TOP_N].min() for v in signed_r_col])


def report_noise_floor_check(name: str, dfa_df: pd.DataFrame, ratio: np.ndarray) -> None:
    floor = top10_floor(dfa_df["signed_vec_R"])
    med = np.median(ratio)
    reliable, diffuse = floor[ratio >= med], floor[ratio < med]
    print(f"\n[{name}] Step 1 — reliability regime vs top-10 floor (median max_mean_ratio={med:.2f}):")
    print(f"  reliable (n={len(reliable)}): min={reliable.min():.4f} p5={np.percentile(reliable,5):.4f} "
          f"median={np.median(reliable):.4f}")
    print(f"  diffuse  (n={len(diffuse)}): min={diffuse.min():.4f} p5={np.percentile(diffuse,5):.4f} "
          f"median={np.median(diffuse):.4f}")


def top10_detail(signed_r: np.ndarray, signed_shuffle: np.ndarray) -> pd.DataFrame:
    """Per-feature detail for a clip's own top-10 by |signed_R|. Noise-first, sign-flip-
    second, decrease/increase residual — locked order (Tom, 05/08). Single source of truth
    for the classification math — classify_clip() aggregates this, nothing recomputes it."""
    top10 = np.argsort(-np.abs(signed_r))[:TOP_N]
    r10, c10 = signed_r[top10], signed_shuffle[top10]
    abs_r10 = np.abs(r10)
    rel_change = (np.abs(c10) - abs_r10) / abs_r10

    is_noise = np.abs(rel_change) <= CHANGE_BAND
    is_sign_flip = ~is_noise & (np.sign(r10) != np.sign(c10))
    residual = ~is_noise & ~is_sign_flip
    bucket = np.where(is_noise, "noise",
              np.where(is_sign_flip, "sign_flip",
              np.where(residual & (rel_change < 0), "decrease", "increase")))
    return pd.DataFrame({"feature_idx": top10, "signed_R": r10, "signed_shuffle": c10, "bucket": bucket})


def classify_clip(signed_r: np.ndarray, signed_shuffle: np.ndarray) -> dict:
    detail = top10_detail(signed_r, signed_shuffle)
    total = detail["signed_R"].abs().sum()
    fracs = detail.groupby("bucket")["signed_R"].apply(lambda s: s.abs().sum() / total)
    return {f"frac_{b}": float(fracs.get(b, 0.0)) for b in ("noise", "sign_flip", "decrease", "increase")}


def build_clip_table(name: str, cfg: dict) -> pd.DataFrame:
    dfa_df = pd.read_parquet(cfg["dfa_parquet"])
    classes = eligible_classes(dfa_df, cfg["r_acc_csv"])
    dfa_df = dfa_df[dfa_df["class_id"].isin(classes)].reset_index(drop=True)

    if cfg["reliability_parquet"] is not None:
        diag = pd.read_parquet(cfg["reliability_parquet"], columns=["clip_id", "max_mean_ratio"])
        ratio = dfa_df.merge(diag, on="clip_id", how="left")["max_mean_ratio"].to_numpy()
    else:
        ratio = max_mean_ratio(dfa_df["signed_vec_R"])
    report_noise_floor_check(name, dfa_df, ratio)

    rows = []
    for _, clip in dfa_df.iterrows():
        r = np.asarray(clip["signed_vec_R"], dtype=np.float32)
        c = np.asarray(clip[cfg["shuffle_col"]], dtype=np.float32)
        row = classify_clip(r, c)
        row["clip_id"] = clip["clip_id"]
        row["class_id"] = clip["class_id"]
        row["correct_under_shuffle"] = bool(clip[cfg["correct_col"]])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_logit(name: str, df: pd.DataFrame) -> None:
    x_cols = ["frac_sign_flip", "frac_decrease", "frac_increase"]  # frac_noise = reference
    x = sm.add_constant(df[x_cols])
    y = df["correct_under_shuffle"].astype(int)
    result = sm.Logit(y, x).fit(disp=0)
    print(f"\n[{name}] Logistic regression — P(correct_under_shuffle) ~ {' + '.join(x_cols)}")
    print(f"  N={len(df)}  base_rate={y.mean():.3f}")
    for term in ["const"] + x_cols:
        print(f"  {term:16s} coef={result.params[term]:+.3f}  se={result.bse[term]:.3f}  "
              f"z={result.tvalues[term]:+.2f}  p={result.pvalues[term]:.4f}")


CSV_COLUMNS = ["clip_id", "class_id", "frac_noise", "frac_sign_flip",
               "frac_decrease", "frac_increase", "correct_under_shuffle"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        print(f"Processing {name}...")
        df = build_clip_table(name, cfg)
        df[CSV_COLUMNS].to_csv(OUT_DIR / f"{name}_clip_shuffle_disruption.csv", index=False)
        print(f"  N={len(df)} clips -> {name}_clip_shuffle_disruption.csv")
        fit_logit(name, df)


if __name__ == "__main__":
    main()
