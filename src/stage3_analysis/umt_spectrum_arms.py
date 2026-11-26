"""26/11/26: spectrum-segment ablation arms -> the positional driver's sets format (feature, arm), one file per layer.
Arms s1..s5 (20% of C75 latents each by 25/11 time-sensitivity pct, s1 = most) x 5% / 10% of the layer's AP mass,
latents by AP within segment (outputs/analysis/pe_groups_251126/arms_mass.csv). Smoke copies for MAX_CLIPS runs.
Usage: MODEL=umt uv run python src/stage3_analysis/umt_spectrum_arms.py"""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = {"model": os.environ.get("MODEL", "umt"), "xs": (0.05, 0.10),
       "arms": ROOT / "outputs/analysis/pe_groups_251126/arms_mass.csv",
       "out": ROOT / "outputs/analysis/{m}_spectrum_ablation"}

A = pd.read_csv(CFG["arms"]).query("model == @CFG['model'] and x in @CFG['xs']")
out = Path(str(CFG["out"]).format(m=CFG["model"])); out.mkdir(parents=True, exist_ok=True)
for L, g in A.groupby("L"):
    s = pd.DataFrame([dict(feature=int(f), arm=f"{r.seg}_{round(r.x * 100)}") for _, r in g.iterrows() for f in r.picks.split()])
    for name in (f"sets_L{L}.csv", f"sets_L{L}_smoke.csv"):
        s.to_csv(out / name, index=False)
    print(f"L{L}:", s.groupby("arm", sort=False).size().to_dict())
