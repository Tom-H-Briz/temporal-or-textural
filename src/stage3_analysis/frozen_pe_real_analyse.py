"""Frozen time PE on real clips (24/11/26; findings/pe_timing_frozen_real_three_backbones_findings_241126.md §3).
Per model and layer, per active latent: order index under normal vs frozen PE (R / Rev vs R_frozen / Rev_frozen),
frame-profile r(R, R_frozen) and clip-total size frozen / normal. Summary: order-sensitive latents (order index > 0.5)
vs the other active latents (control: the freeze can move every latent). CPU, local.
Inputs: outputs/analysis/{model}_layer_pe_characterise/{L, frozen_L}{L}.npz + features_L{L}.csv.
Out: outputs/analysis/frozen_pe_real/{latents_{model}_L{L}.csv, summary.csv}.
Usage: uv run python src/stage3_analysis/frozen_pe_real_analyse.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
from umt_layer_pe_characterise_analyse import frame_corr, order_index

A = ROOT / "outputs/analysis"
CFG = dict(models=("umt", "vm", "tf"), layers=(5, 7, 9), active=0.5, order_cut=0.5, dir=A / "{}_layer_pe_characterise",
           out=A / "frozen_pe_real")


def rows(z, z5, key):
    """Clip row index per input: UMT stores row_<key> at every layer, VM / TF only in the L5 file."""
    return z[f"row_{key}"] if f"row_{key}" in z else z5[f"row_{key}"]


def load_aligned(model, L, cfg):
    """R, Rev, R_frozen, Rev_frozen (clips, frames, F) on the clips present in all four (decode retries can skip)."""
    D = Path(str(cfg["dir"]).format(model))
    z, fz, z5, f5 = (np.load(D / n) for n in (f"L{L}.npz", f"frozen_L{L}.npz", "L5.npz", "frozen_L5.npz"))
    keys = (("R", z, z5), ("Rev", z, z5), ("R_frozen", fz, f5), ("Rev_frozen", fz, f5))
    idx = {k: list(rows(a, b, k)) for k, a, b in keys}
    common = sorted(set.intersection(*map(set, idx.values())))
    return [a[k][[idx[k].index(c) for c in common]] for k, a, _ in keys]


def latent_table(model, L, cfg):
    """One row per active latent: blank-rule group, order index normal / frozen, profile r, size frozen / normal."""
    f = pd.read_csv(Path(str(cfg["dir"]).format(model)) / f"features_L{L}.csv")
    a = f[(f.cls != "dead") & (f.fire_rate >= cfg["active"])]
    R, V, RF, VF = (x[:, :, a.latent.values].astype(np.float32) for x in load_aligned(model, L, cfg))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = pd.DataFrame(dict(latent=a.latent.values, cls=a.cls.values, order_full=order_index(R, V),
                              order_frozen=order_index(RF, VF), profile_r=np.nanmedian(frame_corr(RF, R), 0),
                              size=RF.sum(1).mean(0) / R.sum(1).mean(0)))
    return d.assign(model=model, layer=L, n_clips=len(R), order_sensitive=d.order_full > cfg["order_cut"])


def summarise(d):
    """Order-sensitive vs other active latents: n, median order index normal / frozen, % still locked, profile r, size."""
    g = d.groupby(["model", "layer", "order_sensitive"])
    s = g.agg(n=("latent", "size"), order_full=("order_full", "median"), order_frozen=("order_frozen", "median"),
              profile_r=("profile_r", "median"), size=("size", "median"))
    s["pct_still_locked"] = g.order_frozen.apply(lambda x: 100 * (x > 0.5).mean())
    return s.reset_index()


def main(cfg):
    cfg["out"].mkdir(parents=True, exist_ok=True)
    tabs = []
    for m in cfg["models"]:
        for L in cfg["layers"]:
            d = latent_table(m, L, cfg)
            d.to_csv(cfg["out"] / f"latents_{m}_L{L}.csv", index=False)
            tabs.append(d)
            print(f"{m} L{L}: {d.n_clips.iloc[0]} clips, {int(d.order_sensitive.sum())} order-sensitive", flush=True)
    s = summarise(pd.concat(tabs))
    s.round(4).to_csv(cfg["out"] / "summary.csv", index=False)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
