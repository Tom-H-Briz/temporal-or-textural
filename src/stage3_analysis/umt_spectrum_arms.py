"""26/11/26: spectrum-segment ablation arms -> the positional driver's sets format (feature, arm), one file per layer.
Arms s1..s5 (20% of C75 latents each by time-sensitivity pct, s1 = most) x 5% / 10% of the layer's AP mass, latents by
AP within segment. 27/11: reads the formal spectrum (pe_spectrum/spectrum.parquet, seg + ap columns) instead of the
exploratory pe_groups_251126/arms_mass.csv; the per-layer picking moved here from arms_mass.py. Smoke copies too.
Usage: MODEL=umt uv run python src/stage3_analysis/umt_spectrum_arms.py"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = {"model": os.environ.get("MODEL", "umt"), "xs": (0.05, 0.10),
       "spectrum": ROOT / "outputs/analysis/pe_spectrum/spectrum.parquet",
       "out": ROOT / "outputs/analysis/{m}_spectrum_ablation"}


def pick(h, tot, x):
    """Latents by AP descending until the arm holds x of the layer's AP (arms_mass.py); empty = segment too light."""
    c = h.ap.cumsum().values / tot
    n = int(np.searchsorted(c, x) + 1) if c[-1] >= x else None
    return h.head(n) if n else h.iloc[:0]


def main(model, xs):
    S = pd.read_parquet(CFG["spectrum"]).query("model == @model")
    out = Path(str(CFG["out"]).format(m=model)); out.mkdir(parents=True, exist_ok=True)
    for L, g in S.groupby("L"):
        rows = []
        for s in ("s1", "s2", "s3", "s4", "s5"):
            h = g[g.seg == s].sort_values("ap", ascending=False)
            for x in xs:
                rows += [dict(feature=int(f), arm=f"{s}_{round(x * 100)}") for f in pick(h, g.ap.sum(), x).latent]
        t = pd.DataFrame(rows)
        for name in (f"sets_L{L}.csv", f"sets_L{L}_smoke.csv"):
            t.to_csv(out / name, index=False)
        print(f"L{L}:", t.groupby("arm", sort=False).size().to_dict())


if __name__ == "__main__":
    main(CFG["model"], CFG["xs"])
