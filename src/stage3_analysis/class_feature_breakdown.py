"""
Per-class, cross-backbone feature breakdown — CC brief 05/08 (corrected).

Parameterised (backbone, class_id) tool. Reuses the existing per-clip
classification unchanged (clip_shuffle_disruption.top10_detail) but keeps the
per-feature detail that script only ever aggregated away — which features
drive each bucket, and for sign-flips, what they flip from/to.

Usage:
    uv run python src/stage3_analysis/class_feature_breakdown.py --backbone vm --class-id 6
    uv run python src/stage3_analysis/class_feature_breakdown.py --backbone tf --class-id 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from stage3_analysis.clip_shuffle_disruption import CONFIGS, eligible_classes, top10_detail

OUT_DIR = ROOT / "outputs" / "analysis" / "class_feature_breakdown"

# Post-fix-valid entries only — TF never retrained (pre-fix characterisation still valid),
# VM entries confirmed current this session. See brief for why the list stays this short.
KNOWN_FEATURES = {
    5037: "motion-energy, scatters under shuffle (TF, 23/06)",
    358:  "SSv2 L5 position-locked scaffold member",
    449:  "SSv2 L5 position-locked scaffold member",
    917:  "SSv2 L5 position-locked scaffold member",
    2093: "SSv2 L5 position-locked scaffold member",
    3516: "SSv2 L5 position-locked scaffold member",
    3938: "SSv2 L5 position-locked scaffold member",
    5004: "SSv2 L5 position-locked scaffold member",
}


def build_detail_table(backbone: str, class_id: int) -> pd.DataFrame:
    """One row per (clip, top-10 feature) for this class — R-correct clips only."""
    cfg = CONFIGS[f"ssv2_{backbone}"]
    dfa_df = pd.read_parquet(cfg["dfa_parquet"])
    if class_id not in eligible_classes(dfa_df, cfg["r_acc_csv"]):
        print(f"  WARNING: class {class_id} is not SL-eligible for {backbone} (R-acc < 40% "
              f"or no DFA data) — proceeding anyway since this is a diagnostic tool, not a filter")
    clips = dfa_df[dfa_df["class_id"] == class_id]

    rows = []
    for _, clip in clips.iterrows():
        r = np.asarray(clip["signed_vec_R"], dtype=np.float32)
        c = np.asarray(clip[cfg["shuffle_col"]], dtype=np.float32)
        detail = top10_detail(r, c)
        detail["clip_id"] = clip["clip_id"]
        detail["correct_under_shuffle"] = bool(clip[cfg["correct_col"]])
        rows.append(detail)
    return pd.concat(rows, ignore_index=True)


BUCKETS = ("noise", "sign_flip", "decrease", "increase")


def per_clip_fracs(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (clip, feature) rows back to one row per clip — same fractions classify_clip
    would give, just derived from the retained detail instead of computed standalone."""
    d = detail_df.assign(abs_r=detail_df["signed_R"].abs())
    d["weight"] = d["abs_r"] / d.groupby("clip_id")["abs_r"].transform("sum")
    pivot = d.pivot_table(index="clip_id", columns="bucket", values="weight", aggfunc="sum", fill_value=0.0)
    pivot = pivot.reindex(columns=BUCKETS, fill_value=0.0)
    pivot.columns = [f"frac_{b}" for b in pivot.columns]
    pivot["correct_under_shuffle"] = d.groupby("clip_id")["correct_under_shuffle"].first()
    return pivot.reset_index()


def class_summary(backbone: str, class_id: int, detail_df: pd.DataFrame) -> dict:
    r_acc_df = pd.read_csv(CONFIGS[f"ssv2_{backbone}"]["r_acc_csv"]).set_index("class_id")["accuracy"]
    fracs = per_clip_fracs(detail_df)
    row = {
        "backbone": backbone, "class_id": class_id, "n_clips": len(fracs),
        "R_acc": float(r_acc_df.get(class_id, float("nan"))),
        "shuffle_survival_rate": float(fracs["correct_under_shuffle"].mean()),
    }
    for b in BUCKETS:
        row[f"mean_frac_{b}"] = float(fracs[f"frac_{b}"].mean())
    for flag, label in ((True, "correct"), (False, "incorrect")):
        sub = fracs[fracs["correct_under_shuffle"] == flag]
        for b in BUCKETS:
            row[f"mean_frac_{b}_{label}"] = float(sub[f"frac_{b}"].mean()) if len(sub) else float("nan")
    return row


def feature_breakdown(backbone: str, class_id: int, detail_df: pd.DataFrame) -> pd.DataFrame:
    """Per bucket, features ranked by consistency — the fraction of this class's clips
    where the feature both lands in that clip's own top-10 and is classified into this bucket."""
    n_clips = detail_df["clip_id"].nunique()
    grp = detail_df.groupby(["bucket", "feature_idx"]).agg(
        n_clips_present=("clip_id", "nunique"),
        mean_s_R=("signed_R", "mean"),
        mean_s_shuffle=("signed_shuffle", "mean"),
    ).reset_index()
    grp["frac_of_class_clips"] = grp["n_clips_present"] / n_clips
    grp["backbone"] = backbone
    grp["class_id"] = class_id
    grp["known_as"] = grp["feature_idx"].map(KNOWN_FEATURES).fillna("")
    grp = grp.rename(columns={"feature_idx": "feature_id"})
    cols = ["backbone", "class_id", "bucket", "feature_id", "n_clips_present",
            "frac_of_class_clips", "mean_s_R", "mean_s_shuffle", "known_as"]
    return grp[cols].sort_values(["bucket", "frac_of_class_clips"], ascending=[True, False])


def run_one(backbone: str, class_id: int) -> None:
    print(f"\n=== {backbone} / class {class_id} ===")
    detail = build_detail_table(backbone, class_id)
    summary = class_summary(backbone, class_id, detail)
    breakdown = feature_breakdown(backbone, class_id, detail)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(OUT_DIR / f"{backbone}_class{class_id}_summary.csv", index=False)
    breakdown.to_csv(OUT_DIR / f"{backbone}_class{class_id}_features.csv", index=False)

    print(f"  n_clips={summary['n_clips']}  R_acc={summary['R_acc']:.3f}  "
          f"shuffle_survival_rate={summary['shuffle_survival_rate']:.3f}")
    for b in BUCKETS:
        print(f"  mean_frac_{b}={summary[f'mean_frac_{b}']:.3f}  "
              f"(correct={summary[f'mean_frac_{b}_correct']:.3f}, incorrect={summary[f'mean_frac_{b}_incorrect']:.3f})")
    known_hits = breakdown[breakdown["known_as"] != ""]
    if len(known_hits):
        print("  KNOWN_FEATURES matches:")
        for _, r in known_hits.iterrows():
            print(f"    feature {r['feature_id']} in '{r['bucket']}' ({r['known_as']})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-class, cross-backbone feature breakdown. Omit --backbone to run both, for comparison.",
        epilog="Example: uv run python src/stage3_analysis/class_feature_breakdown.py --class-id 6",
    )
    parser.add_argument("--backbone", choices=["vm", "tf"], help="omit to run both backbones")
    parser.add_argument("--class-id", type=int, required=True)
    args = parser.parse_args()
    backbones = [args.backbone] if args.backbone else ["vm", "tf"]
    for backbone in backbones:
        run_one(backbone, args.class_id)


if __name__ == "__main__":
    main()
