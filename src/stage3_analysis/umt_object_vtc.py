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
L = int(os.environ.get("LAYER", 9))
CFG = dict(oa.CFG, model="umt", layer=L, splice=True, dose=1.0, edit_batch=16,
           num_workers=int(os.environ.get("NUM_WORKERS", 4)),
           out_dir=ROOT / f"outputs/analysis/umt_object_vtc_L{L}_splice")


def clips_arms(cfg):
    """Object test clips (clip_id, true_class, objects named) and arms: batch_base + every edit + control."""
    c = oa.contrast_cfg(dict(cfg, max_clips=0))       # real picks; MAX_CLIPS caps clips only
    s = pd.read_csv(ROOT / "outputs/analysis/umt_object_contrast/sets.csv", dtype={"clip_id": str})
    t = s[s.role == "test"].groupby("clip_id").agg(true_class=("true_class", "first"), objects=("object", ",".join))
    t = t.reset_index().head(cfg["max_clips"] or None)
    return t, [("batch_base", [])] + [a for a in oa.edit_list(c) if a[0] != "baseline"]


def clip_out(st, pv, arms, y, cfg):
    """One clip: every arm's 174 VTC logits (layer L spliced) -> true logit, margin, correct per arm."""
    from umt_vtc_clock_ablation import arm_logits, clip_pass
    _, x, z0 = clip_pass(st, pv)
    lg = torch.stack(arm_logits(st, pv, x, z0, st["D"], arms, cfg))
    other = lg.clone()
    other[:, y] = -torch.inf                          # runner-up = best of the other 173
    return [dict(arm=a, logit_true=float(lg[i, y]), margin=float(lg[i, y] - other[i].max()),
                 correct=bool(lg[i].argmax() == y)) for i, (a, _) in enumerate(arms)]


def run_ablate(cfg):
    """Every test clip (R) -> per_clip.parquet, rewritten every 200 clips so a crash keeps what ran."""
    import vision_pair_phase3
    from umt_mlm_feature_contrast import clip_loader
    clips, arms = clips_arms(cfg)
    st, out, done, t0 = vision_pair_phase3.setup(cfg), [], set(), time.time()
    save = lambda: pd.DataFrame(out).to_parquet(cfg["out_dir"] / "per_clip.parquet", index=False)
    for pv, i in clip_loader(clips, cfg, "R"):
        if int(i) in done:                            # the dataset's decode-retry skip-to-next
            continue
        done.add(int(i))
        r = clips.iloc[int(i)]
        out += [dict(clip_id=r.clip_id, true_class=int(r.true_class), objects=r.objects, **d)
                for d in clip_out(st, pv.to(cfg["device"]), arms, int(r.true_class), cfg)]
        if len(done) % 200 == 0:
            save(); print(f"  {len(done)}/{len(clips)} clips, {(time.time() - t0) / len(done):.2f} s/clip", flush=True)
    save()
    print(f"L{cfg['layer']}: {len(done)}/{len(clips)} clips x {len(arms)} arms in {(time.time() - t0) / 60:.1f} min")


def deltas(cfg):
    """Per clip x edit: edit and its control vs batch_base, side by side; names_obj = caption names the edit's object."""
    df = pd.read_parquet(cfg["out_dir"] / "per_clip.parquet")
    df["correct"] = df.correct.astype(float)
    base = df[df.arm == "batch_base"].set_index("clip_id")[["logit_true", "margin", "correct"]]
    d = df[df.arm != "batch_base"].join(base, on="clip_id", rsuffix="_base")
    for m in ("logit_true", "margin", "correct"):
        d[f"d_{m}"] = d[m] - d[f"{m}_base"]
    d["edit"], d["side"] = d.arm.str.replace("ctrl_", ""), np.where(d.arm.str.startswith("ctrl_"), "ctrl", "edit")
    w = d.pivot_table(index=["clip_id", "edit"], columns="side", values=["d_logit_true", "d_margin", "d_correct"])
    w.columns = [f"{v}_{s}" for v, s in w.columns]
    w = w.join(base.correct.rename("base_correct"), on="clip_id").reset_index()
    objs = df.drop_duplicates("clip_id").set_index("clip_id").objects
    w["names_obj"] = [e.split("_")[0] in objs[c].split(",") for c, e in zip(w.clip_id, w.edit)]
    return w


def run_summarise(cfg):
    """summary_logit.csv (true-class logit and margin, edit - control, CI) and summary_acc.csv (174-way accuracy)."""
    w, lg, acc = deltas(cfg), [], []
    for (e, named), g in w.groupby(["edit", "names_obj"]):
        key = dict(edit=e, clips="names object" if named else "other objects", n=len(g))
        m, lo, hi = oa.boot(g.d_logit_true_edit - g.d_logit_true_ctrl)
        lg.append(dict(key, d_logit=g.d_logit_true_edit.mean(), ctrl=g.d_logit_true_ctrl.mean(), minus_ctrl=m,
                       ci_lo=lo, ci_hi=hi, d_margin_minus_ctrl=(g.d_margin_edit - g.d_margin_ctrl).mean()))
        acc.append(dict(key, base_acc=g.base_correct.mean(), d_acc=g.d_correct_edit.mean(), ctrl_d_acc=g.d_correct_ctrl.mean()))
    for name, rows in (("summary_logit", lg), ("summary_acc", acc)):
        t = pd.DataFrame(rows)
        t.to_csv(cfg["out_dir"] / f"{name}.csv", index=False)
        print(t.round(3).to_string(index=False))


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode, cfg = os.environ.get("MODE", "ablate"), dict(CFG)
    if mode not in ("ablate", "summarise") or os.environ.get("TASK", "object") != "object":
        sys.exit(f"MODE={mode}: use ablate | summarise (object edits only: unset TASK)")
    if cfg["max_clips"]:                              # local checks never overwrite the run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    run_ablate(cfg) if mode == "ablate" else run_summarise(cfg)


if __name__ == "__main__":
    main()
