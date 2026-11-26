"""26/11/26: per-class damage table for the spectrum ablation: per segment s1-s5, logit and top-1 change at 5% / 10%
AP arms, sorted by the 10% top-1 drop. Reads the R-gated summarise tables. Writes class_table.csv next to them.
Usage: RUN_DIR=outputs/analysis/umt_spectrum_ablation_L7_splice/gate_ref uv run python src/stage3_analysis/spectrum_ablation_class_table.py"""
import json, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CFG = {"run_dir": ROOT / os.environ.get("RUN_DIR", "outputs/analysis/umt_spectrum_ablation_L5_splice/gate_ref"),
       "labels": ROOT / "data/ssv2/labels/labels.json", "segments": range(1, 6), "doses": (5, 10)}


def load(cfg):
    """Long table: class, SL label, n, arm, d_logit, top-1 change in points."""
    lg = pd.read_csv(cfg["run_dir"] / "summary_by_class.csv")
    ac = pd.read_csv(cfg["run_dir"] / "summary_by_class_accuracy.csv")
    t = lg.merge(ac[["class_id", "arm", "d_acc"]], on=["class_id", "arm"]).assign(top1=lambda d: 100 * d.d_acc)
    nm = {int(v): k for k, v in json.load(open(cfg["labels"])).items()}
    return t.assign(cls=t.class_id.map(nm))


def segment(t, s, cfg):
    """One segment: one row per class, logit / top-1 at each dose, worst 10% top-1 first."""
    w = t[t.arm.str.startswith(f"s{s}_")].pivot_table(index=["sl_label", "cls", "n"], columns="arm", values=["d_logit", "top1"])
    w = pd.DataFrame({f"{m} {d}%": w[(m, f"s{s}_{d}")] for d in cfg["doses"] for m in ("d_logit", "top1")}).reset_index()
    return w.sort_values([f"top1 {cfg['doses'][-1]}%", f"top1 {cfg['doses'][0]}%"]).assign(segment=f"s{s}")


def main(cfg):
    t = load(cfg); out = pd.concat([segment(t, s, cfg) for s in cfg["segments"]])
    out.round(2).to_csv(cfg["run_dir"] / "class_table.csv", index=False)
    for s, g in out.groupby("segment", sort=False):
        print(f"\n### {s}\n" + g.drop(columns="segment").round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
