"""Clip-bootstrap CIs on the top-20 L5 decompositions of 7 UMT L9 targets (19/11/26), plus the verb-vs-noun gap.
Per target: PE-spectrum share of positive singles, of negative singles (sign option 1), of |all| singles (option 3),
% removed by every arm. Gap: clip-only (targets fixed), hierarchical (targets resampled), exact label permutation.
Inputs: <run>/extract/prof_L{c}_{cond}.npy (umt_top20_extract_targets.py), feature_sheets/ buckets. CPU, local.
Usage: uv run python src/stage3_analysis/umt_top20_bootstrap.py"""
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
CFG = {   # name: (run, capture layer, column in extract/, home class or "obj", feature sheet, kind)
    "down_4250": ("umt_top20_4250_L5_ud_splice", 9, 0, 43, "UMT_4250", "verb"),
    "push_5083": ("umt_top20_push_5083_L5_splice", 9, 0, 93, "UMT_push_5083", "verb"),
    "behind_491": ("umt_top20_behind_491_L5_splice", 9, 0, 8, "UMT_behind_491", "verb"),
    "front_768": ("umt_top20_front_768_L5_splice", 9, 1, 9, "UMT_front_768", "verb"),
    "ball_713": ("umt_top20_ball_713_L5_splice", 9, 0, "obj", "UMT_ball_713", "noun"),
    "paper_4228": ("umt_top20_paper_4228_L5_splice", 9, 0, "obj", "UMT_paper_4228", "noun"),
    "cup_6011": ("umt_top20_cup_6011_L5_splice", 9, 0, "obj", "UMT_cup_6011", "noun"),
}
SPLIT = os.environ.get("SPLIT", "val")   # val | test | pooled (val + held-out test clips, verbs; nouns stay val)
OPT = {"conds": ("R", "Rev"), "n_boot": 2000, "seed": 0, "ci": (2.5, 97.5), "split": SPLIT,
       "out": A / ("umt_top20_bootstrap" + ("" if SPLIT == "val" else f"_{SPLIT}"))}


def clip_totals(run, layer, col, home, cond, split="val"):
    """SPLIT val | test | pooled (19/11/26): stack home-clip totals of run and/or run_test (same arms, asserted).
    No run_test (nouns: test has no object labels) -> val clips in every split."""
    runs = {"val": [run], "test": [run + "_test"], "pooled": [run, run + "_test"]}[split]
    runs = [r for r in runs if (A / r).exists()] or [run]
    got = [one_run(r, layer, col, home, cond) for r in runs]
    assert all(g[1] == got[0][1] for g in got), f"{runs}: arms differ"
    return np.concatenate([g[0] for g in got]), got[0][1]


def one_run(run, layer, col, home, cond):
    """(home clips, arm rows) target activation summed over frames, and arm name -> row."""
    clips = pd.read_csv(A / run / "clips.csv", dtype={"clip_id": str})
    arms = pd.read_csv(A / run / "arms.csv")
    m = clips.role.isin(["pos", "test"]).to_numpy() if home == "obj" else (clips.class_id == home).to_numpy()
    prof = np.load(A / run / "extract" / f"prof_L{layer}_{cond}.npy")[..., col]          # (n, rows, T)
    return prof[m].sum(2), dict(zip(arms.arm, arms.row))


def buckets(sheet):
    """Sheet's 20 features and their PE-spectrum flag (PE driven or mixed = fires on blank)."""
    s = pd.read_csv(A / "feature_sheets" / f"{sheet}_feature_analysis.csv", encoding="utf-8-sig")
    pe = (s["PE driven"] == "x") | (s["Mixed (PE + content)"] == "x")
    return s["Feature Idx"].to_numpy(), pe.to_numpy()


def stats(S, ridx, feats, pe):
    """S: (..., rows) clip-summed target activation. Returns {stat: (...)} with % removed vs batch_base."""
    bb = S[..., [ridx["batch_base"]]]
    pct = 100 * (bb - S) / bb                                                      # (..., rows)
    e = pct[..., [ridx[f"f{f}"] for f in feats]]                                   # single ablations (..., 20)
    pos, neg = np.clip(e, 0, None), np.clip(e, None, 0)
    out = {"pe_share_pos": pos[..., pe].sum(-1) / pos.sum(-1),                    # sign option 1
           "pe_share_neg": neg[..., pe].sum(-1) / neg.sum(-1),
           "pe_share_abs": np.abs(e)[..., pe].sum(-1) / np.abs(e).sum(-1),        # sign option 3
           "pos_sum": pos.sum(-1), "neg_sum": neg.sum(-1)}                        # what each share rests on
    out.update({f"pct_{a}": pct[..., r] for a, r in ridx.items()
                if not a.startswith("f") and a not in ("baseline", "batch_base")})
    return out


def boot_target(T, ridx, feats, pe, rng, n_boot):
    """Resample home clips with replacement; returns point stats and {stat: (n_boot,)} replicates."""
    n = T.shape[0]
    W = rng.multinomial(n, np.full(n, 1 / n), size=n_boot).astype(np.float64)    # (B, n) clip counts
    return stats(T.sum(0), ridx, feats, pe), stats(W @ T, ridx, feats, pe)


def summarise(name, kind, cond, point, reps, ci):
    """One row per stat: point estimate, percentile CI, bootstrap SE."""
    return [{"target": name, "kind": kind, "cond": cond, "stat": s, "point": float(point[s]),
             "lo": np.nanpercentile(reps[s], ci[0]), "hi": np.nanpercentile(reps[s], ci[1]),
             "se": np.nanstd(reps[s])} for s in point]


def gap_tests(point, reps, kinds, rng, ci):
    """Verb mean - noun mean of one stat. point: {target: float}, reps: {target: (B,)}, kinds: {target: verb|noun}."""
    v = [t for t in kinds if kinds[t] == "verb"]
    n = [t for t in kinds if kinds[t] == "noun"]
    obs = np.mean([point[t] for t in v]) - np.mean([point[t] for t in n])
    R = {t: np.asarray(reps[t]) for t in kinds}
    clip = np.mean([R[t] for t in v], 0) - np.mean([R[t] for t in n], 0)          # targets fixed
    B = len(clip)                                                                  # targets resampled too
    hier = np.array([np.mean([R[t][b] for t in rng.choice(v, len(v))]) -
                     np.mean([R[t][b] for t in rng.choice(n, len(n))]) for b in range(B)])
    perm = [np.mean([point[t] for t in s]) - np.mean([point[t] for t in kinds if t not in s])
            for s in combinations(kinds, len(v))]                                  # all 35 labellings
    return {"gap": obs, "clip_lo": np.percentile(clip, ci[0]), "clip_hi": np.percentile(clip, ci[1]),
            "clip_p": np.mean(clip <= 0), "hier_lo": np.percentile(hier, ci[0]),
            "hier_hi": np.percentile(hier, ci[1]), "hier_p": np.mean(hier <= 0),
            "perm_p": np.mean(np.array(perm) >= obs - 1e-12), "n_labellings": len(perm)}


def main(cfg, opt):
    rng, rows, gaps = np.random.default_rng(opt["seed"]), [], []
    opt["out"].mkdir(exist_ok=True)
    for cond in opt["conds"]:
        P, R = {}, {}
        for name, (run, layer, col, home, sheet, kind) in cfg.items():
            T, ridx = clip_totals(run, layer, col, home, cond, opt["split"])
            P[name], R[name] = boot_target(T, ridx, *buckets(sheet), rng, opt["n_boot"])
            rows += summarise(name, kind, cond, P[name], R[name], opt["ci"])
        kinds = {t: v[5] for t, v in cfg.items()}
        for s in ("pe_share_pos", "pe_share_neg", "pe_share_abs"):
            g = gap_tests({t: P[t][s] for t in P}, {t: R[t][s] for t in R}, kinds, rng, opt["ci"])
            gaps.append({"cond": cond, "stat": s, **g})
    pd.DataFrame(rows).round(4).to_csv(opt["out"] / "target_ci.csv", index=False)
    pd.DataFrame(gaps).round(4).to_csv(opt["out"] / "gap_tests.csv", index=False)
    print(pd.DataFrame(gaps).round(3).to_string(index=False))


if __name__ == "__main__":
    main(CFG, OPT)
