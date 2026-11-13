"""
2x2 interaction analysis for umt_clock_downstream_ablation.py runs with TWOBYTWO=1 (CPU, 12/11/26).
Arms: baseline, batch_base, clock, builders, both. Per target and per pair margin:
drop(clock), drop(builders), drop(both) against batch_base, and the interaction
drop(both) - drop(clock) - drop(builders): > 0 super-additive (the gate), < 0 the arms back each other up.
Home clips = the target's home class (None = every clip). rr = ramp_r of the clip-mean frame profile.
Usage: LAYER=5 [RUN_DIR=outputs/analysis/umt_clock_downstream_L5_2x2_ud] uv run python src/stage3_analysis/umt_clock_twobytwo_analyse.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "outputs/analysis"
L = int(os.environ.get("LAYER", 5))
CFG = {
    "layer": L, "capture": [c for c in (7, 9) if c > L],
    "run_dir": Path(os.environ.get("RUN_DIR", str(A / f"umt_clock_downstream_L{L}_2x2_ud"))),
    "arms": ["batch_base", "clock", "builders", "both"], "n_pos": 12, "out": "analysis",
    "pairs": [(45, 43), (168, 165)],
    "targets": {7: {2983: None, 5806: None},
                9: {393: 45, 4250: 43, 1375: 43, 1540: None, 5825: None, 3537: None, 65: None, 4903: None}},
}


def meta(cfg):
    """clips table and arm name -> store row index."""
    clips = pd.read_csv(cfg["run_dir"] / "clips.csv", dtype={"clip_id": str})
    arms = pd.read_csv(cfg["run_dir"] / "arms.csv")
    return clips, dict(zip(arms.arm, arms.row))


def ramp_r(prof_f):
    """ramp_r of one feature's clip-mean frame profile (T,) — correlation with frame index."""
    g = np.asarray(prof_f, dtype=np.float32)
    if not np.isfinite(g).all() or np.ptp(g) == 0:
        return float("nan")
    return float(np.corrcoef(g, np.arange(len(g)))[0, 1])


def target_rows(cfg, clips, ridx, cond):
    """Per capture layer x target: batch_base mass, drop per arm (sum over home clips x frames), interaction,
    share of batch_base, and rr per arm. Interaction = drop(both) - drop(clock) - drop(builders)."""
    rows = []
    for c in cfg["capture"]:
        prof = np.asarray(np.load(cfg["run_dir"] / f"prof_L{c}_{cond}.npy", mmap_mode="r"), dtype=np.float32)
        for t, home in cfg["targets"].get(c, {}).items():
            m = np.ones(len(clips), bool) if home is None else (clips.class_id == home).to_numpy()
            if not m.any():
                continue
            row = {"cond": cond, "layer": c, "feature": t, "home": home, "n_clips": int(m.sum())}
            bb = prof[m, ridx["batch_base"], :, t].sum()
            row["bb"] = bb
            for a in cfg["arms"][1:]:
                row[f"drop_{a}"] = bb - prof[m, ridx[a], :, t].sum()
                row[f"rr_{a}"] = ramp_r(prof[m, ridx[a], :, t].mean(0))
            row["interaction"] = row["drop_both"] - row["drop_clock"] - row["drop_builders"]
            row["pct_clock"], row["pct_builders"], row["pct_both"] = (
                100 * row[f"drop_{a}"] / bb if bb > 1e-3 else float("nan")
                for a in ("clock", "builders", "both"))
            rows.append(row)
    return rows


def margin_rows(cfg, clips, ridx, lg, cond):
    """Per pair side: mean pair margin logit(home) - logit(partner) on home clips, per arm; delta vs batch_base
    and the interaction of the deltas."""
    rows = []
    for a, b in cfg["pairs"]:
        for side, home, part in (("home", a, b), ("partner", b, a)):
            m = (clips.class_id == home).to_numpy()
            if not m.any():
                continue
            row = {"cond": cond, "pair": f"{a}/{b}", "side": side, "n_clips": int(m.sum())}
            for arm in cfg["arms"]:
                row[arm] = float((lg[m, ridx[arm], home] - lg[m, ridx[arm], part]).mean())
            for x in cfg["arms"][1:]:
                row[f"d_{x}"] = row[x] - row["batch_base"]
            row["interaction"] = row["d_both"] - row["d_clock"] - row["d_builders"]
            rows.append(row)
    return rows


def main(cfg):
    out = cfg["run_dir"] / cfg["out"]
    out.mkdir(parents=True, exist_ok=True)
    clips, ridx = meta(cfg)
    tt, mm = [], []
    for cond in ("R", "Rev"):
        lg = np.asarray(np.load(cfg["run_dir"] / f"logits_{cond}.npy"), dtype=np.float32)
        tt += target_rows(cfg, clips, ridx, cond)
        mm += margin_rows(cfg, clips, ridx, lg, cond)
    pd.DataFrame(tt).round(2).to_csv(out / "twobytwo_targets.csv", index=False)
    pd.DataFrame(mm).round(3).to_csv(out / "twobytwo_margin.csv", index=False)
    cols = ["cond", "layer", "feature", "home", "n_clips", "pct_clock", "pct_builders", "pct_both",
            "drop_clock", "drop_builders", "drop_both", "interaction"]
    print(pd.DataFrame(tt)[cols].round(2).to_string(index=False))
    print(pd.DataFrame(mm)[["cond", "pair", "side", "n_clips", "d_clock", "d_builders", "d_both",
                            "interaction"]].round(2).to_string(index=False))
    print(f"written: {out / 'twobytwo_targets.csv'}, {out / 'twobytwo_margin.csv'}")


if __name__ == "__main__":
    main(CFG)
