"""VM down-pool group ablation (19/11/26): which kind of L9 'down' feature carries the evidence? Groups from
vm_synthetic_down_readout + vm_down_cond_readout(_camera): scene (camera-driven), image (object + camera),
object (object only), mixed (the rest). Spliced at L9; group arms, mass-matched controls, the 20 singles.
Clips: 45/43 (up / down) and 168/165 (camera up / down). Usage: [MAX_CLIPS=5] uv run python
src/stage3_analysis/vm_down_group_ablation.py; MODE=summary for the tables."""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
A = ROOT / "outputs/analysis"
GROUPS = dict(scene=[1411, 2686, 5401, 6058], image=[4375, 215, 290, 2888, 4073],
              object=[1825, 2632, 5634, 3936, 5973], mixed=[2391, 2581, 1115, 3254, 4515, 4993])
CFG = dict(model="videomae", layer=9, splice=True, dose=1.0, edit_batch=8, num_workers=0, groups=GROUPS,
           partner={43: 45, 45: 43, 168: 165, 165: 168}, max_other=0.8,      # control pool: down/up mass ratio > 0.8
           max_clips=int(os.environ.get("MAX_CLIPS", 0)), real=A / "vm_clock_downstream_L5_raw",
           video_dir=os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
           device="cuda" if torch.cuda.is_available() else "cpu", out_dir=A / "vm_down_group_ablation_L9_splice")


def controls(cfg):
    """{group: control feats}: per member, the nearest-mass active feature that fires alike on down and up
    (11/11 baseline L9 profiles), excluding all 20; drawn without replacement across groups."""
    from umt_bridge_phase3 import mass_controls
    c = pd.read_csv(cfg["real"] / "clips.csv").class_id.values
    p = np.load(cfg["real"] / "prof_L9_R.npy", mmap_mode="r")
    m = {k: np.asarray(p[c == k, 0], dtype=np.float32).sum(1).mean(0) for k in (43, 45)}
    ratio = np.minimum(m[43], m[45]) / (np.maximum(m[43], m[45]) + 1e-8)
    every = [f for g in cfg["groups"].values() for f in g]
    pool, out = set(np.where((m[43] > 0) & (ratio > cfg["max_other"]))[0].tolist()) - set(every), {}
    for g, feats in list(cfg["groups"].items()) + [("all20", every)]:
        out[g], gap = mass_controls(m[43], np.array(feats), ratio, cfg["max_other"], pool=set(pool))
        if g != "all20":
            pool -= set(out[g].tolist())
        print(f"ctrl {g}: median mass gap {np.median(gap):.3f}", flush=True)
    return {g: v.tolist() for g, v in out.items()}


def arms(cfg, ctrl):
    """[(name, feats)]: batch_base (empty splice, the reference), groups + all20, their controls, 20 singles."""
    every = [f for g in cfg["groups"].values() for f in g]
    a = [("batch_base", [])] + list(cfg["groups"].items()) + [("all20", every)]
    a += [(f"ctrl_{g}", v) for g, v in ctrl.items()]
    return a + [(f"f{f}", [f]) for f in every]


def clip_table(cfg):
    """45/43 + 168/165 clips from the 11/11 run (optionally the first max_clips per class)."""
    c = pd.read_csv(cfg["real"] / "clips.csv")
    c = c[c.class_id.isin(list(cfg["partner"]))]
    if cfg["max_clips"]:
        c = c.groupby("class_id").head(cfg["max_clips"])
    return c.reset_index(drop=True)


def run(cfg):
    """Every clip x arm -> per_clip.parquet (true logit, margin, correct, pair margin), saved every 100 clips."""
    import vision_pair_phase3
    from umt_object_vtc import clip_out
    from vm_object_contrast import clip_loader
    clips, st = clip_table(cfg), vision_pair_phase3.setup(cfg)
    a, out, t0 = arms(cfg, controls(cfg)), [], time.time()
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=n, feats=",".join(map(str, f))) for n, f in a]).to_csv(cfg["out_dir"] / "arms.csv", index=False)
    save = lambda: pd.DataFrame(out).to_parquet(cfg["out_dir"] / "per_clip.parquet", index=False)
    for n, (pv, i) in enumerate(clip_loader(clips, cfg, "R", st["processor"])):
        y = int(clips.class_id.iloc[int(i)])
        out += [dict(clip_id=int(clips.clip_id.iloc[int(i)]), true_class=y, **d)
                for d in clip_out(st, pv.to(cfg["device"]), a, y, cfg["partner"][y], cfg)]
        if (n + 1) % 100 == 0:
            save(); print(f"{n + 1}/{len(clips)} clips, {(time.time() - t0) / (n + 1):.1f} s/clip", flush=True)
    save()


def deltas(cfg):
    """Per arm x true class: mean edit - batch_base of true logit and pair margin; top-1 and pair accuracy in pt."""
    d = pd.read_parquet(cfg["out_dir"] / "per_clip.parquet")
    d["pair_correct"] = d.pair_margin > 0
    b = d[d.arm == "batch_base"].set_index("clip_id")
    for k in ("logit_true", "pair_margin", "correct", "pair_correct"):
        d[f"d_{k}"] = d[k].astype(float).values - b[k].astype(float).reindex(d.clip_id).values
    g = d.groupby(["arm", "true_class"])[["d_logit_true", "d_pair_margin", "d_correct", "d_pair_correct"]].mean()
    g[["d_correct", "d_pair_correct"]] *= 100
    return g.reset_index()


def summary(cfg):
    """Logit table and accuracy table (kept separate), columns = true class; groups / controls first."""
    g = deltas(cfg)
    g.to_csv(cfg["out_dir"] / "summary.csv", index=False)
    order = list(cfg["groups"]) + ["all20"] + [f"ctrl_{k}" for k in list(cfg["groups"]) + ["all20"]]
    order += sorted(set(g.arm) - set(order) - {"batch_base"})
    for cols, name in ((["d_logit_true", "d_pair_margin"], "LOGIT"), (["d_correct", "d_pair_correct"], "ACCURACY (pt)")):
        t = g.pivot(index="arm", columns="true_class", values=cols).reindex(order)
        print(f"\n{name}\n" + t.round(2).to_string())


if __name__ == "__main__":
    summary(CFG) if os.environ.get("MODE") == "summary" else run(CFG)
