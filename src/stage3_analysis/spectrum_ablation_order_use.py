"""26/11/26: does per-class ablation damage track the model's own order use (shuffle drop) or the human SL label?
Per arm, across the SL-35 classes: Spearman rho(damage, shuffle drop) and rho(damage, SL temporal = 1 / static = 0),
damage = -d_acc and -d_logit (R-gated summarise tables). Paired class bootstrap for both rhos and their difference.
Shuffle drop = % of R-correct clips wrong after frame shuffle (correct_C in the all-174 AP file; any layer, same clips).
Usage: RUN_DIR=outputs/analysis/umt_spectrum_ablation_L5_splice/gate_ref uv run python src/stage3_analysis/spectrum_ablation_order_use.py"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
CFG = {"run_dir": ROOT / os.environ.get("RUN_DIR", "outputs/analysis/umt_spectrum_ablation_L5_splice/gate_ref"),
       "ap": ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l9_job7ep_k64.parquet",
       "n_boot": 2000, "seed": 0, "skip_arms": ("baseline", "batch_base")}


def shuffle_drop(cfg):
    """class_id -> % of R-correct clips the model gets wrong once frames are shuffled."""
    d = pd.read_parquet(cfg["ap"], columns=["class_id", "correct_C"])
    return 100 * (1 - d.groupby("class_id").correct_C.mean())


def arm_rows(arm, dmg, shuf, sl, cfg):
    """One arm: rho with shuffle drop, rho with SL label, their difference; paired bootstrap over classes."""
    rng, n = np.random.default_rng(cfg["seed"]), len(dmg)
    rho = lambda i: (spearmanr(dmg[i], shuf[i])[0], spearmanr(dmg[i], sl[i])[0])
    pt = rho(np.arange(n)); bs = np.array([rho(rng.integers(0, n, n)) for _ in range(cfg["n_boot"])])
    bs = np.c_[bs, bs[:, 0] - bs[:, 1]]; ci = np.nanpercentile(bs, [2.5, 97.5], 0)
    return dict(arm=arm, n_classes=n, rho_shuffle=pt[0], rho_shuffle_lo=ci[0, 0], rho_shuffle_hi=ci[1, 0],
                rho_sl=pt[1], rho_sl_lo=ci[0, 1], rho_sl_hi=ci[1, 1],
                diff=pt[0] - pt[1], diff_lo=ci[0, 2], diff_hi=ci[1, 2])


def main(cfg):
    shuf, out = shuffle_drop(cfg), []
    for fname, col in (("summary_by_class_accuracy.csv", "d_acc"), ("summary_by_class.csv", "d_logit")):
        t = pd.read_csv(cfg["run_dir"] / fname)
        for arm, g in t[~t.arm.isin(cfg["skip_arms"])].groupby("arm", sort=False):
            out.append(dict(measure=col, **arm_rows(arm, -g[col].to_numpy(), shuf.reindex(g.class_id).to_numpy(),
                                                    (g.sl_label == "temporal").astype(float).to_numpy(), cfg)))
    o = pd.DataFrame(out); o.to_csv(cfg["run_dir"] / "order_use.csv", index=False)
    print(o.round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
