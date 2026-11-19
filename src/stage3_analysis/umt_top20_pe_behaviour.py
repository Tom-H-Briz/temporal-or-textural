"""What the top-20 L5 inputs of the 7 UMT L9 targets do with the PE (19/11/26). CPU, local.
Per feature x target: bucket (sheet), blank-clip frame behaviour under full / frozen PE (umt_pe_probe: ramp, comb,
frame-shaped, flat), blank / real, single-ablation effect (val R), L5 class breadth (fixed-threshold all-174 DFA).
Writes features.csv, share_by_behaviour.csv (positive ablation mass per behaviour), correlations.csv.
Usage: uv run python src/stage3_analysis/umt_top20_pe_behaviour.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from umt_top20_bootstrap import CFG as TARGETS
from umt_layer_specificity import CFG as SPEC, class_mean, load, spread

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
CFG = {"probe": A / "umt_pe_probe/profiles.npz", "comb_key": ("5", "comb"), "ramp_r": 0.75, "flat_cv": 0.2,
       "sets": Path(__file__).resolve().parent / "umt_clock_downstream_sets.json", "content_lbr": -2.0,
       "out": A / "umt_top20_pe_behaviour"}


def blank_behaviour(P, f, comb, cfg):
    """Blank-clip frame profile of latent f (PE alone): kind, ramp r, CV over frames, frozen / full mass."""
    full, frz = P["blank|full"][:, f], P["blank|frozen"][:, f]
    cv = full.std() / full.mean() if full.mean() > 0 else np.nan
    r = np.corrcoef(full, np.arange(len(full)))[0, 1] if full.std() > 0 else np.nan
    kind = ("comb" if f in comb else "ramp" if abs(r) >= cfg["ramp_r"] and cv >= cfg["flat_cv"]
            else "frame_shaped" if cv >= cfg["flat_cv"] else "flat")
    return {"blank_kind": kind, "ramp_r": r, "cv": cv, "frozen_kept": frz.sum() / full.sum() if full.sum() > 0 else np.nan}


def sheet_rows(name, run, sheet):
    """One row per sheet feature: bucket, blank / real, single-ablation effect (% of target removed, val R)."""
    s = pd.read_csv(A / "feature_sheets" / f"{sheet}_feature_analysis.csv", encoding="utf-8-sig")
    real = next(c for c in s.columns if c.startswith("Real act"))
    g = pd.read_csv(A / run / "analysis/group_ablation.csv").query("cond == 'R'").set_index("arm")
    col = f"pct_L9_{name.split('_')[1]}"
    bucket = np.where(s["PE driven"] == "x", "pe", np.where(s["Mixed (PE + content)"] == "x", "mixed", "content"))
    return pd.DataFrame({"target": name, "feature": s["Feature Idx"], "bucket": bucket,
                         "blank_real": s["Blank act (PE alone)"] / s[real].clip(lower=1e-9),
                         "effect": [g.loc[f"f{f}", col] for f in s["Feature Idx"]]})


def feature_table(cfg):
    """All targets' rows + blank behaviour (content-bucket rows -> 'content') + L5 class breadth (n50)."""
    P = {k: v.astype(np.float64) for k, v in np.load(cfg["probe"]).items()}
    comb = set(json.load(open(cfg["sets"]))[cfg["comb_key"][0]][cfg["comb_key"][1]])
    D = pd.concat([sheet_rows(n, v[0], v[4]).assign(kind=v[5]) for n, v in TARGETS.items()], ignore_index=True)
    D = D.join(pd.DataFrame([blank_behaviour(P, f, comb, cfg) for f in D.feature]))
    D["behaviour"] = np.where(D.bucket == "content", "content", D.blank_kind)
    D["log_blank_real"] = np.log10(D.blank_real.clip(lower=1e-4))
    D["sign"] = np.where(D.effect > 0, "pos", "neg")
    X, cls = load(dict(SPEC, layers=(5,)))
    D["L5_n50"] = spread(class_mean(X[5], cls), D.feature.to_numpy(), 0.5)[0]
    return D


def share_by_behaviour(D):
    """% of each target's positive single-ablation mass per behaviour class."""
    t = D[D.effect > 0].pivot_table(index="target", columns="behaviour", values="effect", aggfunc="sum", fill_value=0)
    return (100 * t.div(t.sum(1), axis=0)).reindex(list(TARGETS))


def correlations(D):
    """log blank / real vs breadth, |effect|, sharing (Spearman; rows and unique features), and sign x kind tests."""
    u = D.drop_duplicates("feature")
    n_t = D.feature.map(D.feature.value_counts())
    rows = [("rows", "L5_n50", *spearmanr(D.log_blank_real, D.L5_n50)), ("unique", "L5_n50", *spearmanr(u.log_blank_real, u.L5_n50)),
            ("rows", "abs_effect", *spearmanr(D.log_blank_real, D.effect.abs())), ("rows", "n_targets", *spearmanr(D.log_blank_real, n_t))]
    out = [{"set": s, "vs": v, "rho": r, "p": p, "n": len(u) if s == "unique" else len(D)} for s, v, r, p in rows]
    for sg in ("pos", "neg"):                                        # verb vs noun inputs, per sign
        v, n = (D[(D.sign == sg) & (D.kind == k)].log_blank_real for k in ("verb", "noun"))
        out.append({"set": f"{sg}: verb vs noun", "vs": "median log blank/real", "rho": v.median() - n.median(),
                    "p": mannwhitneyu(v, n).pvalue, "n": len(v) + len(n)})
    return pd.DataFrame(out)


def main(cfg):
    cfg["out"].mkdir(exist_ok=True)
    D = feature_table(cfg)
    D.round(4).to_csv(cfg["out"] / "features.csv", index=False)
    s, c = share_by_behaviour(D), correlations(D)
    s.round(2).to_csv(cfg["out"] / "share_by_behaviour.csv")
    c.round(4).to_csv(cfg["out"] / "correlations.csv", index=False)
    print(pd.crosstab(D.drop_duplicates("feature").bucket, D.drop_duplicates("feature").blank_kind), "\n")
    print(s.round(1).to_string(), "\n")
    print(D.groupby(["kind", "sign"]).log_blank_real.median().round(2).to_string(), "\n")
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
