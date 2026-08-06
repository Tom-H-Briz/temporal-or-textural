"""
Sign-flip dose-response test — CC brief 06/08 addendum.

Does flip magnitude predict clip failure, not just flip presence? Instance-
level (thousands of sign_flip instances: ~12,450 VM / 1,312 TF), fixing the
32-class power problem — reads ssv2_{vm,tf}_full_detail.parquet directly, no
new extraction. Reports pooled AND class-controlled (class-demeaned swing)
side by side, and raw AND floor-filtered side by side — same lesson as
purity_within_class.py: pooled-only conflates within/between-class effects.

Outputs (outputs/analysis/shuffle_reduction_composition/):
    {backbone}_dose_response_instances.csv — per-instance swing + floor/demean fields
    {backbone}_dose_response_summary.csv   — the 4 raw/floor x pooled/class-demeaned tests

Usage:
    uv run python src/stage3_analysis/signflip_dose_response.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs" / "analysis" / "shuffle_reduction_composition"


def load_signflip(backbone: str) -> pd.DataFrame:
    detail = pd.read_parquet(OUT_DIR / f"ssv2_{backbone}_full_detail.parquet")
    sf = detail[detail["bucket"] == "sign_flip"].copy()
    sf["swing"] = (sf["signed_shuffle"] - sf["signed_R"]).abs()
    return sf


def floor_filter(sf: pd.DataFrame) -> pd.DataFrame:
    """Excludes instances below the class's own median |signed_R| — a flip from
    near-zero to near-zero isn't a meaningful reversal, same noise-boundary risk
    already caught for the sign-bucket classification itself."""
    class_median = sf.groupby("class_id")["signed_R"].transform(lambda s: s.abs().median())
    return sf[sf["signed_R"].abs() >= class_median]


def demean_by_class(sf: pd.DataFrame) -> pd.Series:
    """swing minus its class's own mean swing — removes between-class magnitude
    differences (some classes just swing bigger overall) while preserving each
    instance's position relative to its own class, before pooling for power."""
    return sf["swing"] - sf.groupby("class_id")["swing"].transform("mean")


def mann_whitney_test(swing: pd.Series, correct: pd.Series) -> dict:
    failing, surviving = swing[~correct], swing[correct]
    u = stats.mannwhitneyu(failing, surviving, alternative="two-sided")
    return {
        "n_failing": len(failing), "n_surviving": len(surviving),
        "mean_swing_failing": float(failing.mean()), "mean_swing_surviving": float(surviving.mean()),
        "median_swing_failing": float(failing.median()), "median_swing_surviving": float(surviving.median()),
        "U": float(u.statistic), "p": float(u.pvalue),
    }


def run_backbone(backbone: str) -> None:
    print(f"Processing {backbone}...")
    raw = load_signflip(backbone)
    floored = floor_filter(raw)
    raw["passes_floor"] = raw.index.isin(floored.index)
    raw["swing_demeaned"] = demean_by_class(raw)
    raw.drop(columns=["signed_R", "signed_shuffle", "bucket"]).to_csv(
        OUT_DIR / f"{backbone}_dose_response_instances.csv", index=False)

    versions = {"raw": (raw["swing"], raw["correct_under_shuffle"]),
                "raw_class_demeaned": (raw["swing_demeaned"], raw["correct_under_shuffle"]),
                "floor_filtered": (floored["swing"], floored["correct_under_shuffle"]),
                "floor_filtered_class_demeaned": (demean_by_class(floored), floored["correct_under_shuffle"])}
    rows = []
    for name, (swing, correct) in versions.items():
        result = mann_whitney_test(swing, correct)
        result["version"] = name
        rows.append(result)
        print(f"  [{backbone}] {name}: mean_swing failing={result['mean_swing_failing']:.4f} "
              f"surviving={result['mean_swing_surviving']:.4f}  p={result['p']:.4g}")
    pd.DataFrame(rows)[["version", "n_failing", "n_surviving", "mean_swing_failing",
                        "mean_swing_surviving", "median_swing_failing", "median_swing_surviving",
                        "U", "p"]].to_csv(OUT_DIR / f"{backbone}_dose_response_summary.csv", index=False)


def main() -> None:
    for backbone in ("vm", "tf"):
        run_backbone(backbone)


if __name__ == "__main__":
    main()
