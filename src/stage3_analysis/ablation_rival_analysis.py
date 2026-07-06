"""
Rival class frequency and flip analysis from ablation_margin_view_R.csv.

Outputs (outputs/analysis/scaffold_ablation/):
    ablation_rival_frequency.csv   — per true_class × rank2_class
    ablation_flip_clips.csv        — clips where margin_r1r2_abl < 0

Usage:
    uv run python src/stage3_analysis/ablation_rival_analysis.py
"""

from pathlib import Path
import pandas as pd

ROOT    = Path(__file__).parent.parent.parent
RUN_TAG = "cover_uncover_060726"   # change for each new run — stamps source and output filenames
OUT     = ROOT / "outputs/analysis/scaffold_ablation"
SRC     = OUT / f"ablation_margin_view_R_{RUN_TAG}.csv"


def rival_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["flipped"] = df["margin_r1r2_abl"] < 0
    grp = df.groupby(["true_class", "rank2_class"])
    result = grp.agg(
        n_appearances=("clip_id", "count"),
        n_flips=("flipped", "sum"),
        mean_margin_r1r2_pct_change=("margin_r1r2_pct_change", "mean"),
    ).reset_index()
    result["flip_rate"] = result["n_flips"] / result["n_appearances"]
    cols = ["true_class", "rank2_class", "n_appearances", "n_flips",
            "flip_rate", "mean_margin_r1r2_pct_change"]
    return result[cols].sort_values("flip_rate", ascending=False).reset_index(drop=True)


def flip_clips(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["clip_id", "true_class", "rank2_class",
            "margin_r1r2_unabl", "margin_r1r2_abl", "margin_r1r2_pct_change"]
    return (df[df["margin_r1r2_abl"] < 0][cols]
            .sort_values("margin_r1r2_abl")
            .reset_index(drop=True))


def main() -> None:
    df = pd.read_csv(SRC)
    print(f"  {len(df)} clips loaded")

    freq  = rival_frequency(df)
    flips = flip_clips(df)

    freq.to_csv(OUT / f"ablation_rival_frequency_{RUN_TAG}.csv", index=False)
    flips.to_csv(OUT / f"ablation_flip_clips_{RUN_TAG}.csv", index=False)

    print(freq.to_string(index=False))
    print()
    print(f"{len(flips)} flip clips (margin_r1r2_abl < 0):")
    print(flips.to_string(index=False))


if __name__ == "__main__":
    main()
