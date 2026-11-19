"""Joins vm_down_cond_readout (real R / C1 / Rev) to vm_synthetic_down_readout (synthetic tuning), per feature.
Synthetic group: silent (best synthetic < 25% of real down), down (down index > 0.3), else other.
Usage: uv run python src/stage3_analysis/vm_down_cond_analyse.py"""
import os
from pathlib import Path
import numpy as np, pandas as pd

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
CFG = dict(cond=A / "vm_down_cond_readout", synth=A / "vm_synthetic_down_readout/summary.csv",
           silent_cut=0.25, down_cut=0.3, out=A / "vm_down_cond_readout/summary.csv")


def group(s, cfg):
    """Synthetic group label per feature row."""
    if s.best_synth_vs_real < cfg["silent_cut"]:
        return "silent"
    return "down" if s.down_idx > cfg["down_cut"] else "other"


def shape_r(a, b):
    """Pearson r between two mean tubelet profiles (8,)."""
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def per_feature(p, cls):
    """Per feature: masses (clip sum, mean over clips) per class x cond, and R-vs-C1 shape r on down clips."""
    rows = []
    for j, f in enumerate(p["latents"]):
        m = {f"{k}_{c}": p[k][cls == cid][:, :, j].sum(1).mean()
             for k in ("R", "C1", "Rev") for c, cid in (("down", 43), ("up", 45))}
        d = cls == 43
        m["shape_r_C1"] = shape_r(p["R"][d][:, :, j].mean(0), p["C1"][d][:, :, j].mean(0))
        rows.append(dict(latent=int(f), **m))
    return pd.DataFrame(rows).set_index("latent")


def main(cfg):
    p = np.load(cfg["cond"] / "profiles.npz")
    cls = pd.read_csv(cfg["cond"] / "clips.csv").class_id.values
    s, m = pd.read_csv(cfg["synth"]).set_index("latent"), per_feature(p, cls)
    s["group"] = s.apply(group, axis=1, cfg=cfg)
    t = s[["group", "passed", "pref_dir", "down_idx", "best_synth_vs_real"]].join(m)
    t["C1/R"], t["Rev/R"] = t.C1_down / t.R_down, t.Rev_down / t.R_down          # down clips
    t["up/down"], t["RevUp/R_up"] = t.R_up / t.R_down, t.Rev_up / t.R_up        # up clips; Rev of up = down motion
    t["RevUp/down"] = t.Rev_up / t.R_down                                       # does reversed up reach down's level?
    t = t.sort_values(["group", "Rev/R"])
    t.to_csv(cfg["out"])
    cols = ["group", "passed", "pref_dir", "best_synth_vs_real", "R_down", "C1/R", "shape_r_C1", "Rev/R",
            "up/down", "RevUp/R_up", "RevUp/down"]
    print(t[cols].round(2).to_string())
    print(t.groupby("group")[cols[3:]].median().round(2).to_string())


def camera(cfg):
    """Camera up (168, scene moves down) / down (165) vs object down (43), per feature (CAMERA=1)."""
    t = pd.read_csv(cfg["out"]).set_index("latent")[["group", "R_down"]]
    p = np.load(cfg["cond"].with_name("vm_down_cond_readout_camera") / "profiles.npz")
    cls = pd.read_csv(cfg["cond"].with_name("vm_down_cond_readout_camera") / "clips.csv").class_id.values
    mass = lambda k, c: p[k][cls == c].sum(1).mean(0)                          # (n_feats,)
    t["camUp/objDown"] = mass("R", 168) / t.R_down.values
    t["C1/R camUp"], t["Rev/R camUp"] = mass("C1", 168) / mass("R", 168), mass("Rev", 168) / mass("R", 168)
    t["camDown/camUp"] = mass("R", 165) / mass("R", 168)
    t["RevCamDown/camUp"] = mass("Rev", 165) / mass("R", 168)                  # reversed camera down = scene moves down
    print(t.sort_values(["group", "camUp/objDown"]).round(2).to_string())
    print(t.drop(columns="R_down").groupby("group").median(numeric_only=True).round(2).to_string())


if __name__ == "__main__":
    camera(CFG) if os.environ.get("CAMERA") == "1" else main(CFG)
