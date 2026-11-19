"""Analyse the frozen-PE group ablation (19/11/26; umt_frozen_pe_groups.py). CPU, local.
Per layer and arm (group, group x PE pool, all): latents, activation kept under frozen PE, DFA mass kept (+ / - split),
correct-class logit change when only that arm's latents take their frozen-PE values (clip-bootstrap 95% CI), accuracy change.
Out: outputs/analysis/umt_frozen_pe_groups/summary.csv. Usage: uv run python src/stage3_analysis/umt_frozen_pe_groups_analyse.py"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/analysis/umt_frozen_pe_groups"
CFG = {"dir": D, "layers": (5, 7, 9), "groups": Path(__file__).with_name("umt_order_groups.json"),
       "n_boot": 2000, "seed": 0, "out": D / "summary.csv"}


def boot_ci(v, n, rng):
    """95% percentile CI of the mean of v over clip resamples."""
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def arm_row(z, y, a, idx, lat, cfg, rng):
    """One arm: activation / DFA kept over its latents, patched correct-logit and accuracy change vs full PE."""
    act = lambda k: z[k][:, :, lat].astype(np.float64).sum()
    pos = lambda k: np.clip(z[k][:, lat], 0, None).sum()
    neg = lambda k: -np.clip(z[k][:, lat], None, 0).sum()
    c = np.arange(len(y))
    base, pat = z["full_logits"], z["patched"][:, idx]
    dl = pat[c, y] - base[c, y]
    lo, hi = boot_ci(dl, cfg["n_boot"], rng)
    return {"arm": a, "n_lat": len(lat), "act_kept": act("frozen_act") / act("full_act"),
            "pos_dfa_kept": pos("frozen_dfa") / pos("full_dfa"), "neg_dfa_kept": neg("frozen_dfa") / neg("full_dfa"),
            "dlogit": dl.mean(), "dlogit_lo": lo, "dlogit_hi": hi,
            "dacc_pt": 100 * ((pat.argmax(1) == y).mean() - (base.argmax(1) == y).mean())}


def layer_table(L, cfg, rng):
    z, sets = dict(np.load(cfg["dir"] / f"L{L}.npz")), json.loads(cfg["groups"].read_text())[str(L)]
    y = pd.read_csv(cfg["dir"] / "clips.csv").class_id.to_numpy()[z["row"]]
    rows = [arm_row(z, y, a, i, sets.get(a, list(range(6144))), cfg, rng) for i, a in enumerate(z["arms"])]
    return pd.DataFrame(rows).assign(layer=L)


if __name__ == "__main__":
    rng = np.random.default_rng(CFG["seed"])
    t = pd.concat([layer_table(L, CFG, rng) for L in CFG["layers"]])
    t.round(4).to_csv(CFG["out"], index=False)
    print(t.round(3).to_string(index=False))
