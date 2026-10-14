"""Ablation impact of the pair-differentiating features vs their mass-matched controls (and,
where the run has them, the k largest |DFA| contributors as a maximum-damage reference), from
Phase 3's raw data: accuracy (true beats partner: baseline T > 0) and logit margin, for sets
(k = 1, 5, 10), per pair, and per single feature, at dose 1 (ablate) and 2 (flip).
Works on UMT Phase 3 (targets T_vtm, T_match) and the classifier port (T_cls): every
per_clip_interventions.parquet under --dir is read with its sibling per_clip_ap.parquet.
Usage: python src/stage3_analysis/phase3_ablation_impact.py [--dir outputs/analysis/vision_pair_phase3/videomae_L9]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

CFG = {
    "dir": ROOT / "outputs/analysis/umt_vision_text_bridge/phase3",   # default: UMT Phase 3
    "pair_names": {(6, 171): "cover/uncover", (46, 5): "open/close", (14, 172): "fold/unfold",
                   (168, 165): "camera up/down", (45, 43): "move up/down",
                   (44, 41): "towards/away camera", (42, 40): "closer to/away from",
                   (37, 36): "closer/apart"},
    "kind_names": {"set": "selected", "control_set": "control", "max_set": "max |DFA|",
                   "single": "selected", "control_single": "control"},
}


def load(d, cfg):
    """All intervention records under d joined to their clips' baselines (base_<target>)."""
    parts = []
    for f in sorted(Path(d).rglob("per_clip_interventions.parquet")):
        base = pd.read_parquet(f.parent / "per_clip_ap.parquet")
        cols = ["clip_id"] + [c for c in base.columns if c.startswith("base_")]
        parts.append(pd.read_parquet(f).merge(base[cols], on="clip_id"))
    df = pd.concat(parts, ignore_index=True)
    df["pair"] = [cfg["pair_names"].get((a, b), f"{a}_{b}") for a, b in zip(df.a, df.b)]
    df["set"] = df.kind.map(cfg["kind_names"])
    return df


def impact(g, tgt):
    """One group of edits on target tgt: pair accuracy before / after (T > 0), share of
    baseline-correct clips flipped to wrong, and the margin change (mean, median, % of base)."""
    base, delta = g[f"base_{tgt}"], g[f"delta_{tgt}"]
    after, ok = base + delta, base > 0
    return pd.Series({"n": len(g), "acc_base": ok.mean(), "acc_after": (after > 0).mean(),
                      "d_acc": (after > 0).mean() - ok.mean(),
                      "correct_to_wrong": ((after <= 0) & ok).sum() / max(ok.sum(), 1),
                      "margin_base": base.mean(), "d_margin": delta.mean(),
                      "d_margin_median": delta.median(), "pct_margin_removed": -delta.mean() / base.mean()})


def table(df, tgt, kinds, by):
    """impact() per (by..., set, k, dose) for edits selected by target tgt."""
    s = df[(df.target_src == tgt) & df.kind.isin(kinds)]
    return s.groupby(by + ["set", "k", "dose"]).apply(lambda g: impact(g, tgt)).reset_index()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=CFG["dir"])
    d = p.parse_args().dir
    df = load(d, CFG)
    sets, singles, per_feat = ["set", "control_set", "max_set"], ["single", "control_single"], []
    for tgt in sorted(df.target_src.unique()):
        overall = table(df, tgt, sets, [])
        overall.to_csv(d / f"ablation_impact_overall_{tgt}.csv", index=False)
        table(df, tgt, sets, ["pair"]).to_csv(d / f"ablation_impact_by_pair_{tgt}.csv", index=False)
        per_feat.append(table(df, tgt, singles, ["pair", "feats"]).assign(target=tgt))
        print(f"\n== {tgt}: sets, selected vs control ==\n", overall.round(3).to_string(index=False))
    pd.concat(per_feat).to_csv(d / "ablation_impact_per_feature.csv", index=False)
    print(f"\nwritten to {d}: ablation_impact_overall_*.csv, ablation_impact_by_pair_*.csv, "
          "ablation_impact_per_feature.csv")


if __name__ == "__main__":
    main()
