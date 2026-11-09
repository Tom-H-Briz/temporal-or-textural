"""Do UMT's class predictions (VTC, 174-way) use the object features? (05/11/26)
Each layer's object edits (umt_object_contrast_L{LAYER} picks: singles, groups, mass-matched
controls) spliced in at layer L on every held-out object test clip (R); VTC logits read out.
Per clip x arm: true-class logit, margin to the runner-up class, correct -> per_clip.parquet.
Modes: ablate (GPU) | summarise (CPU). Usage: LAYER=7 MODE=ablate MAX_CLIPS=2 uv run python src/stage3_analysis/umt_object_vtc.py
"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
import umt_object_ablation as oa
L, TASK = int(os.environ.get("LAYER", 9)), os.environ.get("TASK", "object")   # TASK=pair: verb-pair edits
MODEL = os.environ.get("MODEL", "umt")               # videomae: its classifier, picks from vm_object_contrast.py
CFG = dict(oa.CFG, model=MODEL, layer=L, splice=True, dose=1.0, edit_batch=16,
           num_workers=int(os.environ.get("NUM_WORKERS", 4)),
           out_dir=ROOT / f"outputs/analysis/{MODEL}_{TASK}_vtc_L{L}_splice")


def clips_arms(cfg):
    """Object test clips (clip_id, true_class, objects named) and arms: batch_base + every edit + control."""
    c = oa.contrast_cfg(dict(cfg, max_clips=0))       # real picks; MAX_CLIPS caps clips only
    s = pd.read_csv(ROOT / f"outputs/analysis/umt_{TASK}_contrast/sets.csv", dtype={"clip_id": str})
    t = s[s.role == "test"].groupby("clip_id").agg(true_class=("true_class", "first"), objects=("object", ",".join))
    t = t.reset_index().head(cfg["max_clips"] or None)
    # pairs: the partner class (45_43 on a 45 clip -> 43); objects: none (-1)
    alt = lambda o, y: [int(c) for c in o.split("_") if int(c) != y][0]
    t["alt_class"] = [alt(o, y) if TASK == "pair" else -1 for o, y in zip(t.objects, t.true_class)]
    return t, [("batch_base", [])] + [a for a in oa.edit_list(c) if a[0] != "baseline"]


def clip_out(st, pv, arms, y, alt, cfg):
    """One clip: every arm's 174 VTC logits (layer L spliced) -> true logit, margin, correct per arm;
    pairs (alt >= 0) also pair_margin = true - partner logit."""
    from umt_vtc_clock_ablation import arm_logits, clip_pass
    _, x, z0 = clip_pass(st, pv)
    lg = torch.stack(arm_logits(st, pv, x, z0, st["D"], arms, cfg))
    other = lg.clone()
    other[:, y] = -torch.inf                          # runner-up = best of the other 173
    return [dict(arm=a, logit_true=float(lg[i, y]), margin=float(lg[i, y] - other[i].max()),
                 correct=bool(lg[i].argmax() == y), pair_margin=float(lg[i, y] - lg[i, alt]) if alt >= 0 else np.nan)
            for i, (a, _) in enumerate(arms)]


def run_ablate(cfg):
    """Every test clip (R) -> per_clip.parquet, rewritten every 200 clips so a crash keeps what ran."""
    import vision_pair_phase3
    from umt_mlm_feature_contrast import clip_loader
    clips, arms = clips_arms(cfg)
    st, out, done, t0 = vision_pair_phase3.setup(cfg), [], set(), time.time()
    save = lambda: pd.DataFrame(out).to_parquet(cfg["out_dir"] / "per_clip.parquet", index=False)
    if cfg["model"] == "videomae":                    # VM's own sampler + processor
        from vm_object_contrast import clip_loader as vm_loader
        loader = vm_loader(clips, cfg, "R", st["processor"])
    else:
        loader = clip_loader(clips, cfg, "R")
    for pv, i in loader:
        if int(i) in done:                            # the dataset's decode-retry skip-to-next
            continue
        done.add(int(i))
        r = clips.iloc[int(i)]
        out += [dict(clip_id=r.clip_id, true_class=int(r.true_class), objects=r.objects, **d)
                for d in clip_out(st, pv.to(cfg["device"]), arms, int(r.true_class), int(r.alt_class), cfg)]
        if len(done) % 200 == 0:
            save(); print(f"  {len(done)}/{len(clips)} clips, {(time.time() - t0) / len(done):.2f} s/clip", flush=True)
    save()
    print(f"L{cfg['layer']}: {len(done)}/{len(clips)} clips x {len(arms)} arms in {(time.time() - t0) / 60:.1f} min")


M = ["logit_true", "margin", "correct"] + (["pair_margin", "pair_correct"] if TASK == "pair" else [])


def clip_group(objs, y, e):
    """Objects: does the caption name the edit's object. Pairs: home class / partner class / other pairs."""
    if TASK == "object":
        return "names object" if e.split("_")[0] in objs.split(",") else "other objects"
    pair, home = "_".join(e.split("_")[:2]), int(e.split("_")[2])
    return "other pairs" if pair != objs else "home class" if y == home else "partner class"


def deltas(cfg):
    """Per clip x edit: edit and its control vs batch_base, side by side, plus the clip's group for that edit."""
    df = pd.read_parquet(cfg["out_dir"] / "per_clip.parquet")
    df["correct"], df["pair_correct"] = df.correct.astype(float), (df.pair_margin > 0).astype(float)
    base = df[df.arm == "batch_base"].set_index("clip_id")[M]
    d = df[df.arm != "batch_base"].join(base, on="clip_id", rsuffix="_base")
    for m in M:
        d[f"d_{m}"] = d[m] - d[f"{m}_base"]
    d["edit"], d["side"] = d.arm.str.replace("ctrl_", ""), np.where(d.arm.str.startswith("ctrl_"), "ctrl", "edit")
    w = d.pivot_table(index=["clip_id", "edit"], columns="side", values=[f"d_{m}" for m in M])
    w.columns = [f"{v}_{s}" for v, s in w.columns]
    w = w.join(base.add_prefix("base_"), on="clip_id").reset_index()
    info = df.drop_duplicates("clip_id").set_index("clip_id")
    w["group"] = [clip_group(info.objects[c], info.true_class[c], e) for c, e in zip(w.clip_id, w.edit)]
    return w


def run_summarise(cfg):
    """summary_logit.csv (true-class logit and margin, edit - control, CI) and summary_acc.csv (174-way accuracy)."""
    w, lg, acc = deltas(cfg), [], []
    for (e, grp), g in w.groupby(["edit", "group"]):
        key = dict(edit=e, clips=grp, n=len(g))
        m, lo, hi = oa.boot(g.d_logit_true_edit - g.d_logit_true_ctrl)
        lg.append(dict(key, d_logit=g.d_logit_true_edit.mean(), ctrl=g.d_logit_true_ctrl.mean(), minus_ctrl=m,
                       ci_lo=lo, ci_hi=hi, d_margin_minus_ctrl=(g.d_margin_edit - g.d_margin_ctrl).mean()))
        acc.append(dict(key, base_acc=g.base_correct.mean(), d_acc=g.d_correct_edit.mean(), ctrl_d_acc=g.d_correct_ctrl.mean()))
        if TASK == "pair":                            # within-pair: true vs partner logit
            lg[-1]["d_pair_margin_minus_ctrl"] = (g.d_pair_margin_edit - g.d_pair_margin_ctrl).mean()
            acc[-1].update(base_pair_acc=g.base_pair_correct.mean(), d_pair_acc=g.d_pair_correct_edit.mean(),
                           ctrl_d_pair_acc=g.d_pair_correct_ctrl.mean())
    for name, rows in (("summary_logit", lg), ("summary_acc", acc)):
        t = pd.DataFrame(rows)
        t.to_csv(cfg["out_dir"] / f"{name}.csv", index=False)
        print(t.round(3).to_string(index=False))


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode, cfg = os.environ.get("MODE", "ablate"), dict(CFG)
    if mode not in ("ablate", "summarise") or TASK not in ("object", "pair"):
        sys.exit(f"MODE={mode} TASK={TASK}: use ablate | summarise, TASK object | pair")
    if cfg["max_clips"]:                              # local checks never overwrite the run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    run_ablate(cfg) if mode == "ablate" else run_summarise(cfg)


if __name__ == "__main__":
    main()
