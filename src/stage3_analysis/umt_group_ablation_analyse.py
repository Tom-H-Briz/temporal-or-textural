"""Group + single-feature ablation analysis (18/11/26) for umt_clock_downstream_ablation.py ARM_GROUPS runs.
Per arm: % of each target's clip-summed activation removed (home clips only, vs batch_base) and the change in the
pair margin logit(home) - logit(partner) on each side's clips. R and Rev. CPU.
Usage: RUN_DIR=outputs/analysis/umt_group_ablation_L5_ud_splice uv run python src/stage3_analysis/umt_group_ablation_analyse.py"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from umt_clock_twobytwo_analyse import meta

A = Path(__file__).resolve().parents[2] / "outputs/analysis"
CFG = {
    "run_dir": Path(os.environ.get("RUN_DIR", str(A / "umt_group_ablation_L5_ud_splice"))),
    "targets": {9: {4250: 43, 393: 45}},      # capture layer -> {target latent: home class}
    "pairs": [(45, 43)],                      # (up, down)
    "readout_cls": 43,                        # class readout on the target's home clips
    "labels": Path(__file__).resolve().parents[2] / "data/ssv2/labels/labels.json",
    "out": "analysis",
}
if os.environ.get("TARGETS"):       # 18/11/26: TARGETS=layer:latent[:home class],...; nouns (NOUN set): home = "obj"
    noun = bool(os.environ.get("NOUN"))  # = the noun's pos + test clips; verbs: PAIRS=a,b[;c,d] READOUT=<class>
    CFG.update(targets={}, readout_cls="obj" if noun else int(os.environ.get("READOUT", 43)),
               pairs=[tuple(map(int, p.split(","))) for p in os.environ.get("PAIRS", "").split(";") if p])
    for t in os.environ["TARGETS"].split(","):
        c, f, *h = t.split(":")
        CFG["targets"].setdefault(int(c), {})[int(f)] = "obj" if noun else int(h[0])


def home_mask(clips, home):
    """Clips a target / readout is measured on: a class id, or 'obj' = object clips (role pos / test)."""
    return clips.role.isin(["pos", "test"]).to_numpy() if home == "obj" else (clips.class_id == home).to_numpy()


def target_pct(cfg, clips, ridx, cond):
    """{f'pct_L{c}_{t}': (arms,)} % of target t's batch_base activation (home clips x frames) removed per arm."""
    out = {}
    for c, tg in cfg["targets"].items():
        prof = np.load(cfg["run_dir"] / f"prof_L{c}_{cond}.npy", mmap_mode="r")
        for t, home in tg.items():
            m = np.where(home_mask(clips, home))[0]
            tot = np.asarray(prof[:, :, :, t], dtype=np.float32)[m].sum((0, 2))          # (rows,)
            bb = tot[ridx["batch_base"]]
            out[f"pct_L{c}_{t}"] = 100 * (bb - tot) / bb
    return out


def margin_delta(cfg, clips, ridx, cond):
    """{f'dmargin_{home}': (arms,)} change in mean margin logit(home) - logit(partner) on home clips vs batch_base."""
    lg, out = np.load(cfg["run_dir"] / f"logits_{cond}.npy").astype(np.float32), {}
    for a, b in cfg["pairs"]:
        for home, part in ((a, b), (b, a)):
            m = (clips.class_id == home).to_numpy()
            mg = (lg[m, :, home] - lg[m, :, part]).mean(0)                               # (rows,)
            out[f"dmargin_{home}"] = mg - mg[ridx["batch_base"]]
    return out


def class_readout(cfg, clips, ridx, cond):
    """Per arm, on readout_cls clips: mean logit change of all 174 classes vs batch_base. 'supports' = class whose
    logit falls most when the arm is ablated; 'suppresses' = class whose logit rises most."""
    import json
    lp = os.environ.get("LABELS_PATH", str(cfg["labels"]))
    names = {int(v): k for k, v in json.load(open(lp)).items()}
    lg = np.load(cfg["run_dir"] / f"logits_{cond}.npy").astype(np.float32)
    m = home_mask(clips, cfg["readout_cls"])
    d = (lg[m] - lg[m][:, [ridx["batch_base"]]]).mean(0)                                  # (rows, 174)
    lo, hi = d.argmin(1), d.argmax(1)
    return {"supports": [names[c] for c in lo], "supports_dlogit": d[np.arange(len(d)), lo],
            "suppresses": [names[c] for c in hi], "suppresses_dlogit": d[np.arange(len(d)), hi]}


def main(cfg):
    clips, ridx = meta(cfg)
    arms = pd.read_csv(cfg["run_dir"] / "arms.csv").sort_values("row")
    out, tabs = cfg["run_dir"] / cfg["out"], []
    out.mkdir(exist_ok=True)
    for cond in ("R", "Rev"):
        t = arms[["arm", "k"]].assign(cond=cond, **target_pct(cfg, clips, ridx, cond), **margin_delta(cfg, clips, ridx, cond),
                                      **class_readout(cfg, clips, ridx, cond))
        tabs.append(t[~t.arm.isin(["baseline", "batch_base"])])
    tab = pd.concat(tabs)
    tab.round(3).to_csv(out / "group_ablation.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
