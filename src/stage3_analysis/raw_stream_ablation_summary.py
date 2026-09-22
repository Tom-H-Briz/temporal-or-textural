"""
Summary of raw_stream_feature_ablation.py's per-clip parquet.

Per (ablation_target x mode): mean logit delta, 95% CI, one-sample p, flip rate.
Deltas are paired within clip against that arm's own base point — raw_* against
the untouched forward pass, recon_* against the unablated SAE splice — so the
SAE reconstruction cost cancels out of every number here.

Population is filtered to raw-correct clips for ALL arms, not just the raw ones:
"R-correct" upstream means correct with the reconstruction spliced in, so a raw
arm evaluated on that population would re-admit the SAE through the sampling.

Outputs:
    outputs/analysis/raw_stream_ablation/raw_stream_ablation_summary_l5_n{n}.csv

Usage:
    uv run python src/stage3_analysis/raw_stream_ablation_summary.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

CFG = {
    "in_path": ROOT / "outputs/analysis/raw_stream_ablation/raw_stream_ablation_l5_n500.parquet",
    "out_dir": ROOT / "outputs/analysis/raw_stream_ablation",
    "layer": 5,
}
INTERVENTION = {"minus": "subtract z_i*d_i", "project": "project out span(d_i)"}


def load_paired(in_path: Path) -> tuple[pd.DataFrame, int, int]:
    """Attach each intervention row to its own arm's base-point logit, and
    restrict to clips the unmodified model already gets right."""
    df = pd.read_parquet(in_path)
    anchor = df[df["mode"].isin(["raw", "recon"])].pivot(
        index="clip_id", columns="mode", values=["correct_class_logit", "correct"])
    keep = anchor[anchor[("correct", "raw")]].index
    d = df[df.clip_id.isin(keep) & (df.ablation_target != "none")].copy()
    d["base"] = np.where(
        d["mode"].str.startswith("raw"),
        d.clip_id.map(anchor[("correct_class_logit", "raw")]).astype(float),
        d.clip_id.map(anchor[("correct_class_logit", "recon")]).astype(float))
    d["delta"] = d["base"].astype(float) - d["correct_class_logit"].astype(float)
    return d, df.clip_id.nunique(), len(keep)


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, mode), g in d.groupby(["ablation_target", "mode"]):
        v = g["delta"].values.astype(float)
        se = v.std(ddof=1) / np.sqrt(len(v))
        _, p = stats.ttest_1samp(v, 0)
        base, kind = mode.split("_")
        rows.append({
            "ablation_target": target, "base_point": base,
            "intervention": INTERVENTION[kind], "n_clips": len(v),
            "mean_delta_logit": v.mean(), "ci95_lo": v.mean() - 1.96 * se,
            "ci95_hi": v.mean() + 1.96 * se, "p_value": p,
            "flip_rate": 1 - g["correct"].mean(),
        })
    return pd.DataFrame(rows).sort_values(
        ["ablation_target", "base_point", "intervention"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_parquet", type=Path, nargs="?", default=CFG["in_path"],
                        help="raw_stream_feature_ablation.py output; defaults to the n500 local run")
    args = parser.parse_args()
    d, n_sampled, n_kept = load_paired(args.results_parquet)
    summary = summarise(d)
    out = CFG["out_dir"] / f"raw_stream_ablation_summary_l{CFG['layer']}_n{n_sampled}.csv"
    summary.to_csv(out, index=False)
    print(f"{n_sampled} clips sampled, {n_kept} raw-correct")
    print(summary[summary.ablation_target == "all7"].to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
