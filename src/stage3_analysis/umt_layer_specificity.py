"""Do UMT features get more class-specific from L5 to L9? (19/11/26). Fixed-threshold all-174 DFA, clips common to
L5 / L7 / L9. |DFA| (grad x z of the correct-class logit, clip-summed) averaged per class -> M (174, 6144) per layer.
A: latents holding 50% of each class's mass.  B: classes served by each feature in any class's top 20.
C: per-feature class spread (top-5 share, n classes for 50%).  D: the 7 L9 targets vs their top-20 L5 parents.
Usage: uv run python src/stage3_analysis/umt_layer_specificity.py"""
from pathlib import Path
import numpy as np
import pandas as pd

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
CFG = {
    "dfa": A / "dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l{}_job7ep_k64.parquet",
    "layers": (5, 7, 9), "top_k": 20, "mass": 0.5, "n_boot": 500, "seed": 0,
    "targets": {"down_4250": "UMT_4250", "push_5083": "UMT_push_5083", "behind_491": "UMT_behind_491",
                "front_768": "UMT_front_768", "ball_713": "UMT_ball_713", "paper_4228": "UMT_paper_4228",
                "cup_6011": "UMT_cup_6011"},
    "out": A / "umt_layer_specificity",
}


def load(cfg):
    """{layer: (clips, 6144) |DFA| float32} on clips present at every layer, and their class ids (same order)."""
    d = {L: pd.read_parquet(str(cfg["dfa"]).format(L), columns=["clip_id", "class_id", "signed_vec_R"])
         for L in cfg["layers"]}
    common = sorted(set.intersection(*[set(x.clip_id) for x in d.values()]))
    X = {L: np.abs(np.stack(x.set_index("clip_id").loc[common, "signed_vec_R"].to_numpy())).astype(np.float32)
         for L, x in d.items()}
    return X, d[cfg["layers"][0]].set_index("clip_id").loc[common, "class_id"].to_numpy()


def class_mean(X, cls):
    """(174, F) mean |DFA| per class (equal weight per class from here on)."""
    return pd.DataFrame(X).groupby(cls).mean().to_numpy()


def n_for_mass(P, mass):
    """Rows of P (each sums to 1): how many largest entries reach `mass`."""
    return (np.cumsum(-np.sort(-P, 1), 1) < mass).sum(1) + 1


def measures(M, cfg):
    """A (per class), B and C (per feature in any class's top k) from class-mean matrix M (174, F)."""
    a = n_for_mass(M / M.sum(1, keepdims=True), cfg["mass"])                           # A: latents per class
    top = np.argsort(-M, 1)[:, :cfg["top_k"]]                                          # (174, k) features that matter
    feats, n_cls = np.unique(top, return_counts=True)                                  # B: classes served per feature
    P = (M[:, feats] / M[:, feats].sum(0, keepdims=True)).T                            # (feats, 174) class spread
    top5 = -np.sort(-P, 1)[:, :5].sum(1)
    return {"A_latents_for_half": a, "B_classes_served": n_cls, "B_distinct_feats": np.array([len(feats)]),
            "C_top5_share": top5, "C_classes_for_half": n_for_mass(P, cfg["mass"])}, feats


def spread(M, idx, mass):
    """Classes holding `mass` of each listed feature's |DFA| (n50), and its top-5 class share."""
    P = (M[:, idx] / M[:, idx].sum(0, keepdims=True)).T
    return n_for_mass(P, mass), -np.sort(-P, 1)[:, :5].sum(1)


def targets_vs_parents(Ms, cfg):
    """D: one row per target: L9 target n50 / top-5 vs the median of its 20 L5 parents (and how many are narrower)."""
    rows = []
    for name, sheet in cfg["targets"].items():
        t = int(name.split("_")[1])
        par = pd.read_csv(A / "feature_sheets" / f"{sheet}_feature_analysis.csv", encoding="utf-8-sig")["Feature Idx"]
        (tn,), (t5,) = spread(Ms[9], [t], cfg["mass"])
        pn, p5 = spread(Ms[5], par.to_numpy(), cfg["mass"])
        rows.append({"target": name, "L9_n50": tn, "L5_parents_n50_median": np.median(pn), "parents_narrower": (pn < tn).sum(),
                     "L9_top5": t5, "L5_parents_top5_median": np.median(p5)})
    return pd.DataFrame(rows)


def summary(Ms, cfg):
    """One flat {stat: value} per resample: layer medians of A / C, mean B, distinct count, D per target."""
    s = {}
    for L, M in Ms.items():
        m, _ = measures(M, cfg)
        s.update({f"L{L}_{k}": float(np.mean(v) if k.startswith("B") else np.median(v)) for k, v in m.items()})
    d = targets_vs_parents(Ms, cfg).set_index("target")
    s.update({f"D_{t}_{c}": float(d.loc[t, c]) for t in d.index for c in ("L9_n50", "L5_parents_n50_median")})
    return s


def resample(X, cls, rng):
    """Stratified clip bootstrap: clips redrawn within each class; returns {layer: class-mean matrix}."""
    idx = np.concatenate([rng.choice(np.where(cls == c)[0], (cls == c).sum()) for c in np.unique(cls)])
    return {L: class_mean(x[idx], cls[idx]) for L, x in X.items()}


def main(cfg):
    X, cls = load(cfg)
    rng, cfg["out"] = np.random.default_rng(cfg["seed"]), Path(cfg["out"])
    cfg["out"].mkdir(exist_ok=True)
    point = summary({L: class_mean(x, cls) for L, x in X.items()}, cfg)
    reps = pd.DataFrame([summary(resample(X, cls, rng), cfg) for _ in range(cfg["n_boot"])])
    for a, b in ((5, 9), (5, 7), (7, 9)):                                      # layer differences, per resample
        for k in ("A_latents_for_half", "B_classes_served", "B_distinct_feats", "C_top5_share", "C_classes_for_half"):
            point[f"L{b}-L{a}_{k}"] = point[f"L{b}_{k}"] - point[f"L{a}_{k}"]
            reps[f"L{b}-L{a}_{k}"] = reps[f"L{b}_{k}"] - reps[f"L{a}_{k}"]
    t = pd.DataFrame({"point": pd.Series(point), "lo": reps.quantile(0.025), "hi": reps.quantile(0.975)})
    t.round(4).to_csv(cfg["out"] / "specificity_ci.csv")
    print(f"{len(cls)} clips common to L5/L7/L9, {cfg['n_boot']} stratified resamples")
    print(t.round(3).to_string())


if __name__ == "__main__":
    main(CFG)
