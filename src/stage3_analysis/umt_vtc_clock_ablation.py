"""UMT L9 clock ablation under the VTC 174-way head, SL subset (brief 211026).

Which classes rely on UMT's 7 "clock" L9 SAE features (early 5473/2417/3948/5043, late
6059/2261/3623) and the 6 positionally consistent core features (4788/1282/1347/4301/1694/541;
fixed frame profiles kept under Rev / C) — as groups, alone, and all 14 core (+ 6127, content)?
Ablates every arm (dose=1 raw-stream edit, umt_bridge_phase3.apply_edit) on all clips of
the 35 SL classes (Static_Temporal.csv x manifest cap40) — the clocks as group arms AND
each alone — and stores ALL 174 logits per clip x arm (fp16), so every summary runs on CPU.
Every delta is measured against `batch_base`, an empty edit inside the batched forward, so
the B=1 baseline vs B=16 kernel offset (~1e-5) cancels; the `baseline` arm stores that
offset. Group arms get mass-matched control sets drawn outside the core 14 and the
46-feature MLM pool. Stages (MODE): mass | ablate | summarise | compare | check | all
(check needs mass first).
Usage: MODE=ablate MAX_CLIPS=5 uv run python src/stage3_analysis/umt_vtc_clock_ablation.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

CFG = {
    "layer": int(os.environ.get("LAYER", 9)), "sae_k": 64, "job_label": "7ep",
    "model": "umt", "dose": 1.0,                       # ablation only (apply_edit's default is 2)
    "early": [5473, 2417, 3948, 5043], "late": [6059, 2261, 3623],
    # the rest of the core 14 (22 Oct): fixed multi-peak frame profiles, kept under Rev / C
    # ("positionally consistent", not ramps); 6127 = the one content-following core feature
    "positional": [4788, 1282, 1347, 4301, 1694, 541],
    # every clock and positional feature alone too (the redundancy test), plus 6127
    "singles": {f"single_{f}": f for f in (5473, 2417, 3948, 5043, 6059, 2261, 3623,
                                           4788, 1282, 1347, 4301, 1694, 541, 6127)},
    "conds": os.environ.get("CONDS", "R"),             # R first (user, 21 Oct); C/Rev = second pass
    "sl_csv": ROOT / "outputs/Laura_SL/Static_Temporal.csv",
    "manifest": ROOT / "outputs/manifests/manifest_ssv2_all174_cap40.json",
    "core14": ROOT / "outputs/analysis/umt_core_ubiquitous/core14.csv",
    "mlm_pool46": ROOT / "outputs/analysis/umt_core_ubiquitous/mlm_shared_pool_46.csv",
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "out_dir": Path(os.environ.get("OUT_DIR", str(ROOT / "outputs/analysis/umt_vtc_clock_ablation"))),
    "max_clips": int(os.environ.get("MAX_CLIPS", 0)), "edit_batch": 16,
    "num_workers": int(os.environ.get("NUM_WORKERS", 4)),   # 0 = no loader workers (no /dev/shm use)
    "n_boot": 1000, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
CFG["all_clocks"] = CFG["early"] + CFG["late"]
LG = [f"lg_{j:03d}" for j in range(174)]               # zero-padded: pivot keeps logits ordered


def sl_classes(cfg):
    """The 35 SL classes: class_id, template, sl_label. One template sits in both columns;
    first occurrence wins (temporal first), which reproduces the brief's 18/17 split."""
    norm = lambda s: s.lower().replace("[", "").replace("]", "")
    key2id = {norm(k): int(v) for k, v in json.load(open(cfg["labels_path"])).items()}
    sl = pd.read_csv(cfg["sl_csv"])
    rows, kept = [], {}
    for label, col in (("temporal", "SL - temporal"), ("static", "SL - static")):
        for tpl in sl[col].dropna():
            if norm(tpl) in kept:
                print(f"SL overlap: '{tpl}' is in both columns; kept as {kept[norm(tpl)]}")
                continue
            kept[norm(tpl)] = label
            rows.append(dict(class_id=key2id[norm(tpl)], template=tpl, sl_label=label))
    return pd.DataFrame(rows).sort_values("class_id").reset_index(drop=True)


def clip_table(cfg):
    """Every manifest clip of the SL classes, all clips not R-correct only (user, 21 Oct):
    clip_id (str), class_id, sl_label; sorted, capped by MAX_CLIPS for smoke runs."""
    norm = lambda s: s.lower().replace("[", "").replace("]", "")
    cls = sl_classes(cfg).assign(key=lambda d: d.template.map(norm)).set_index("key")
    man = [c for c in json.load(open(cfg["manifest"]))["all"] if norm(c["template"]) in cls.index]
    df = pd.DataFrame(man).rename(columns={"id": "clip_id"})
    df["class_id"] = df.template.map(norm).map(cls.class_id)
    df["sl_label"] = df.template.map(norm).map(cls.sl_label)
    df = df[["clip_id", "class_id", "sl_label"]].astype({"clip_id": str})
    return df.sort_values("clip_id").head(cfg["max_clips"] or None).reset_index(drop=True)


def setup(cfg):
    """vision_pair_phase3's setup, unchanged: UMT with the VTC head, the L9 SAE at its
    trained threshold, dim_mean and the layer-L splice hook."""
    import vision_pair_phase3
    return vision_pair_phase3.setup(dict(cfg))


def full_logits(st, pv, new):
    """(B, 174) VTC logits with `new` (B, N, D) spliced at layer L (None = plain forward):
    forward_T's batching without its two-class contrast, so margins and 'lean' are computable."""
    B = 1 if new is None else new.shape[0]
    st["new"] = new
    try:
        out = st["model"](pixel_values=pv.expand(B, *pv.shape[1:])).logits
    finally:
        st["new"] = None
    return out.float()


def clip_pass(st, pv):
    """One clip: baseline logits (174,), the layer-L residual x9 (1, N, D) from the hook,
    and its SAE codes z0 (N, C) — x9 cached once per clip (brief §Arms)."""
    with torch.no_grad():
        base = full_logits(st, pv, None)[0]
        x9 = st["x"].float()
        _, z0 = st["sae"].encode((x9[0] - st["dm"]).float())
    return base, x9, z0


def arm_rows(cfg):
    """[(arm, feats)] in fixed order: the batch-1 baseline, `batch_base` (an empty edit
    inside the batched forward — the reference every delta is measured against), the 3
    group arms (early / late / all clocks, positional, core14), the 14 singles, then the
    control arms read back from arms.csv (clock groups only; the rest have no mass peers)."""
    arms = [("baseline", []), ("batch_base", []), ("early", cfg["early"]),
            ("late", cfg["late"]), ("all", cfg["all_clocks"]),
            ("positional", cfg["positional"]),                      # no control: no mass peers
            ("core14", cfg["all_clocks"] + cfg["positional"] + [6127])]
    arms += [(name, [f]) for name, f in cfg["singles"].items()]
    p = cfg["out_dir"] / "arms.csv"
    assert p.exists(), "arms.csv missing: run MODE=mass first (it draws the controls)"
    a = pd.read_csv(p)
    c = a[a.control_features.notna() & (a.control_features != "")]
    return arms + [(arm, [int(x) for x in feats.split(",")])
                   for arm, feats in zip(c.control_arm, c.control_features)]


def arm_logits(st, pv, x9, z0, D, edits, cfg):
    """The non-baseline arms for one clip, edit_batch per spliced forward (eval_edits'
    batching, no grad); dose is passed explicitly — apply_edit's default is 2."""
    from umt_bridge_phase3 import apply_edit
    lgs = []
    with torch.no_grad():
        for i in range(0, len(edits), cfg["edit_batch"]):
            new = torch.stack([apply_edit(x9, z0, D, fs, dose=cfg["dose"])[0]
                               for _, fs in edits[i:i + cfg["edit_batch"]]])
            lgs += list(full_logits(st, pv, new))
    return lgs


def group_arm_rows(cfg, mass):
    """Group arms + one mass-matched control per member (group_controls' matching via
    umt_bridge_phase3.mass_controls: nearest mass, heaviest first, no replacement; shared
    pool = active features outside the core 14, the MLM 46 and the ablated features)."""
    from umt_bridge_phase3 import mass_controls
    excluded = set(cfg["all_clocks"] + list(cfg["singles"].values())) \
        | set(pd.read_csv(cfg["core14"]).feature.astype(int)) \
        | set(pd.read_csv(cfg["mlm_pool46"]).feature.astype(int))
    pool = set(np.where(mass > 0)[0].tolist()) - excluded
    rows = []
    for arm, feats in (("early", cfg["early"]), ("late", cfg["late"]), ("all", cfg["all_clocks"])):
        ctrl, gap = mass_controls(mass, np.array(feats), None, None, pool=pool)
        pool -= set(ctrl.tolist())
        rows.append(dict(arm=arm, kind="group", features=",".join(map(str, feats)), k=len(feats),
                         mass=float(mass[feats].mean()), control_arm=f"ctrl_{arm}",
                         control_features=",".join(map(str, ctrl.tolist())),
                         control_mass=float(mass[ctrl].mean()), mass_gap=float(gap.max())))
    return rows


def write_arms(cfg, mass):
    """Stage 1 outputs: mass_R.npy + arms.csv (one row per arm; singles carry no control —
    user decision, 21 Oct)."""
    rows = group_arm_rows(cfg, mass)
    rows += [dict(arm=name, kind="single", features=str(f), k=1, mass=float(mass[f]),
                  control_arm="", control_features="", control_mass=np.nan, mass_gap=np.nan)
             for name, f in cfg["singles"].items()]
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "arms.csv", index=False)
    np.save(cfg["out_dir"] / "mass_R.npy", mass)
    g = pd.DataFrame(rows)
    print(g[["arm", "features", "control_features", "mass_gap"]].to_string(index=False))
    print(f"worst group-arm mass gap: {g[g.kind == 'group'].mass_gap.max():.3f}")


def clip_rows(r, cond, arms, base, lgs):
    """One clip's per-arm rows: clip fields + the full 174-logit vector each; base_correct
    flags the batch-1 baseline argmax (the population filter). Logits -> CPU once each."""
    cls = int(r.class_id)
    head = dict(clip_id=r.clip_id, class_id=cls, sl_label=r.sl_label, condition=cond,
                base_correct=int(base.argmax() == cls))
    vals = [base.cpu().tolist()] + [lg.cpu().tolist() for lg in lgs]
    return [dict(head, arm=arm, **dict(zip(LG, v)))
            for arm, v in zip([a for a, _ in arms], vals)]


def run_mass(cfg, st):
    """Stage 1 (GPU): mean per-feature SAE activation mass over the SL clips (R), then the
    control draw. One forward per clip; the loader's skip-to-next is detectable by index."""
    from umt_mlm_feature_contrast import clip_loader
    rows, total, n, done, t0 = clip_table(cfg), None, 0, set(), time.time()
    for pv, i in clip_loader(rows, cfg, "R"):
        if int(i) in done:
            continue                                  # the dataset's skip-to-next: already counted
        done.add(int(i))
        _, _, z0 = clip_pass(st, pv.to(cfg["device"]))
        s = z0.sum(0).cpu().numpy()
        total = s if total is None else total + s
        n += 1
        if n % 200 == 0:
            print(f"  mass {n}/{len(rows)} clips, {(time.time() - t0) / n:.2f} s/clip", flush=True)
    write_arms(cfg, total / max(n, 1))
    print(f"mass stage: {len(done)}/{len(rows)} clips in {(time.time() - t0) / 60:.1f} min")


def ablate_cond(cfg, st, arms, edits, D, cond, out):
    """One condition: every SL clip -> baseline + all arm rows appended to `out`."""
    from umt_mlm_feature_contrast import clip_loader
    from umt_vision_pair_contrast import apply_condition
    rows, done, t0, off = clip_table(cfg), set(), time.time(), 0.0
    bb = [a for a, _ in arms].index("batch_base") - 1     # edits list: baseline is dropped
    for pv, i in clip_loader(rows, cfg, "C" if cond == "C" else "R"):
        if int(i) in done:
            continue                                  # the dataset's skip-to-next: already done
        done.add(int(i))
        r = rows.iloc[int(i)]
        pv = apply_condition(pv, cond).to(cfg["device"])
        base, x9, z0 = clip_pass(st, pv)
        lgs = arm_logits(st, pv, x9, z0, D, edits, cfg)
        off = max(off, float((base - lgs[bb]).abs().max()))   # batch-1 vs batched kernel offset
        out += clip_rows(r, cond, arms, base, lgs)
    print(f"  {cond}: {len(done)}/{len(rows)} clips in {(time.time() - t0) / 60:.1f} min, "
          f"worst |batch1 - batch_base| {off:.2e}", flush=True)


def run_ablate(cfg, st):
    """Stage 2 (GPU): baseline + every arm per clip x condition -> per_clip_logits.parquet
    (174 fp16 logits per row), rewritten after each condition so a crash keeps what ran."""
    arms = arm_rows(cfg)
    edits = [(a, f) for a, f in arms if a != "baseline"]
    D = st["sae"].get_dictionary()
    out = []
    for cond in cfg["conds"].split("+"):
        ablate_cond(cfg, st, arms, edits, D, cond, out)
        df = pd.DataFrame(out)
        df[LG] = df[LG].astype(np.float16)
        df.to_parquet(cfg["out_dir"] / "per_clip_logits.parquet", index=False)
    print(f"ablate: {len(out)} clip-arm rows -> per_clip_logits.parquet")


def load_logits(cfg):
    """per_clip_logits.parquet -> (per-clip table, logits L (n, A, 174) float32, arm names)."""
    df = pd.read_parquet(cfg["out_dir"] / "per_clip_logits.parquet")
    df["key"] = df.clip_id + "|" + df.condition
    arms = list(dict.fromkeys(df.arm))
    piv = df.pivot(index="key", columns="arm", values=LG)
    piv.columns = piv.columns.swaplevel(0, 1)
    piv = piv.sort_index(axis=1)
    cl = df[df.arm == arms[0]].set_index("key").loc[piv.index].reset_index()
    L = np.stack([piv[a].to_numpy().astype(np.float32) for a in arms], 1)
    return cl, L, arms


def measures(L, y, ref):
    """Per-clip x arm arrays vs arm `ref` (batch_base, so the batch-kernel offset cancels):
    d true-class logit, d margin to the runner-up class, d 174-way accuracy (edit argmax
    minus reference argmax, as 0/1)."""
    n, idx = len(y), np.arange(len(y))
    true = L[idx, :, y]
    other = L.copy()
    other[idx, :, y] = -np.inf                     # mask the true class: runner-up = max of the rest
    d_true = true - true[:, [ref]]
    d_margin = (true - other.max(-1)) - (true[:, ref] - other[:, ref].max(-1))[:, None]
    correct = (L.argmax(-1) == y[:, None]).astype(np.float32)
    return d_true, d_margin, correct - correct[:, [ref]]


def add_control_arms(meas, arms):
    """Append each group arm's control-subtracted arrays (paired per clip) as extra arms
    '<arm>_vs_ctrl', so every downstream table covers them unchanged."""
    d_true, d_margin, d_acc = meas
    for g in ("early", "late", "all"):
        i, j = arms.index(g), arms.index(f"ctrl_{g}")
        d_true = np.c_[d_true, d_true[:, i] - d_true[:, j]]
        d_margin = np.c_[d_margin, d_margin[:, i] - d_margin[:, j]]
        d_acc = np.c_[d_acc, d_acc[:, i] - d_acc[:, j]]
        arms = arms + [f"{g}_vs_ctrl"]
    return (d_true, d_margin, d_acc), arms


def ci_mean(v, idx):
    """Mean with a bootstrap 95% CI over the clip draws idx."""
    return float(v.mean()), *[float(x) for x in np.percentile(v[idx].mean(1), [2.5, 97.5])]


def class_table(meas, y, arms, cl, cfg):
    """summary_by_class.csv: per class x arm, mean d logit and d margin, bootstrap CI over
    clips (seed 0, umt_mlm_group_ablation.resample_idx)."""
    from umt_mlm_group_ablation import resample_idx
    rows = []
    for c in np.unique(y):
        m = y == c
        idx = resample_idx(int(m.sum()), cfg["n_boot"], cfg["seed"])
        for j, arm in enumerate(arms):
            row = dict(class_id=int(c), sl_label=cl.sl_label[m].iloc[0], arm=arm, n=int(m.sum()))
            for key, arr in (("d_logit", meas[0]), ("d_margin", meas[1])):
                row[key], row[f"{key}_lo"], row[f"{key}_hi"] = ci_mean(arr[m, j], idx)
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(cfg["out_dir"] / "summary_by_class.csv", index=False)
    return tab


def acc_table(meas, y, arms, cl, cfg):
    """summary_by_class_accuracy.csv: per class x arm, d 174-way accuracy + CI — a separate
    table from the logit/margin one (brief §Measures)."""
    from umt_mlm_group_ablation import resample_idx
    rows = []
    for c in np.unique(y):
        m = y == c
        idx = resample_idx(int(m.sum()), cfg["n_boot"], cfg["seed"])
        for j, arm in enumerate(arms):
            d, lo, hi = ci_mean(meas[2][m, j], idx)
            rows.append(dict(class_id=int(c), sl_label=cl.sl_label[m].iloc[0], arm=arm,
                             n=int(m.sum()), d_acc=d, d_acc_lo=lo, d_acc_hi=hi))
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "summary_by_class_accuracy.csv", index=False)


def sl_gap(cls_tab, value, n_boot, seed):
    """temporal - static difference of one per-class value column, bootstrap over classes
    (resampled within each label) -> per-arm label means, diff and CI."""
    piv = cls_tab.pivot_table(index="class_id", columns="arm", values=value)
    lab = cls_tab.drop_duplicates("class_id").set_index("class_id").loc[piv.index].sl_label
    v, l = piv.to_numpy(), lab.to_numpy()
    ti, si = np.where(l == "temporal")[0], np.where(l == "static")[0]
    rng = np.random.default_rng(seed)
    draws = np.stack([v[rng.choice(ti, len(ti)), :].mean(0) - v[rng.choice(si, len(si)), :].mean(0)
                      for _ in range(n_boot)])
    out = pd.DataFrame(dict(arm=piv.columns, mean_temporal=v[ti].mean(0),
                            mean_static=v[si].mean(0), diff=v[ti].mean(0) - v[si].mean(0)))
    out["lo"], out["hi"] = np.percentile(draws, [2.5, 97.5], axis=0)
    return out.assign(measure=value)


def lean_table(L, arms):
    """lean.csv (long): per arm x class, the mean logit change over ALL clips vs batch_base,
    with each clip's mean change over the 174 classes removed first — VTC logits share a
    common mode, so 'lean' must be a relative gain, not a global shift."""
    d = L - L[:, [arms.index("batch_base")]]
    d = d - d.mean(-1, keepdims=True)
    d = pd.DataFrame(d.mean(0), index=arms, columns=pd.Index(range(L.shape[2]), name="class_id"))
    return d.rename_axis("arm").reset_index().melt(id_vars="arm", var_name="class_id",
                                                   value_name="d_logit_rel")


def run_summarise(cfg):
    """Stage 3 (CPU): per-class tables, summary_sl.csv + summary_sl_accuracy.csv (temporal
    vs static, control-subtracted arms included), lean.csv."""
    cl, L, arms = load_logits(cfg)
    y = cl.class_id.to_numpy()
    meas, arms_full = add_control_arms(measures(L, y, arms.index("batch_base")), arms)
    ct = class_table(meas, y, arms_full, cl, cfg)
    acc_table(meas, y, arms_full, cl, cfg)
    at = pd.read_csv(cfg["out_dir"] / "summary_by_class_accuracy.csv")
    pd.concat([sl_gap(ct, v, cfg["n_boot"], cfg["seed"]) for v in ("d_logit", "d_margin")],
              ignore_index=True).to_csv(cfg["out_dir"] / "summary_sl.csv", index=False)
    sl_gap(at, "d_acc", cfg["n_boot"], cfg["seed"]).to_csv(
        cfg["out_dir"] / "summary_sl_accuracy.csv", index=False)
    lean_table(L, arms).to_csv(cfg["out_dir"] / "lean.csv", index=False)
    print(ct.groupby("arm")[["d_logit", "d_margin"]].mean().round(3).to_string())
    print("summarise done: summary_by_class(.accuracy).csv, summary_sl(.accuracy).csv, lean.csv")


def compare_pair(name, dmg, lab, other):
    """One UMT-vs-VideoMAE row: Spearman of per-class damage, top-10 damaged overlap, and
    the temporal-static gap on each side (rankings, not raw sizes)."""
    from scipy.stats import spearmanr
    top = lambda s: set(s.nlargest(10).index)
    gap = lambda s: float(s[lab.to_numpy() == "temporal"].mean() - s[lab.to_numpy() == "static"].mean())
    return dict(comparison=name, n=len(dmg), spearman=float(spearmanr(dmg, other).statistic),
                top10_overlap=len(top(dmg) & top(other)),
                umt_temporal_static=gap(dmg), vm_temporal_static=gap(other))


def run_compare(cfg):
    """Post-fix stage: UMT damage = the control-subtracted all-clocks logit drop per class;
    VideoMAE L5 = scaffold_dose_raw (alpha 0 vs 1) and ablation_l5_vs_l7's mean_delta_l5
    (a logit drop, so positive = damaged already). No *clean7* files (brief §Measures)."""
    ct = pd.read_csv(cfg["out_dir"] / "summary_by_class.csv")
    u = ct[ct.arm == "all_vs_ctrl"].assign(dmg=lambda d: -d.d_logit).set_index("class_id")
    raw = pd.read_parquet(ROOT / "outputs/analysis/scaffold_dose_raw/scaffold_dose_raw_results.parquet")
    m = raw.pivot_table(index="class_id", columns="alpha", values="logit")
    d2 = pd.read_csv(ROOT / "outputs/analysis/scaffold_ablation/"
                     "ablation_l5_vs_l7_class_impact_job7ep.csv").set_index("class_id").mean_delta_l5
    j = u[["sl_label", "dmg"]].join((m[1.0] - m[0.0]).rename("dmg_dose_raw")).join(
        d2.rename("dmg_l5l7")).dropna()
    rows = [compare_pair("dose_raw_alpha0_vs_alpha1", j.dmg, j.sl_label, j.dmg_dose_raw),
            compare_pair("l5_vs_l7_mean_delta", j.dmg, j.sl_label, j.dmg_l5l7)]
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "compare_videomae.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def check_clip(cfg, st, ref, D, rows, i, pv):
    """One clip: baseline-vs-hook-free |diff| and a dose=0 edit vs baseline (allclose at
    1e-5 — same batch size here, but kernels need not match bitwise). Returns both diffs."""
    from umt_bridge_phase3 import apply_edit
    with torch.no_grad():
        clean = ref(pixel_values=pv).logits.float()[0]
    base, x9, z0 = clip_pass(st, pv)
    d0 = apply_edit(x9, z0, D, cfg["all_clocks"], dose=0.0)
    diff = float((base - clean).abs().max())
    diff0 = float((base - full_logits(st, pv, d0)[0]).abs().max())
    assert diff0 < 1e-5, f"dose=0 differs from baseline by {diff0:.2e}"
    print(f"  clip {rows.iloc[int(i)].clip_id}: baseline |diff| {diff:.2e}, dose=0 {diff0:.2e}")
    return diff, diff0


def run_check(cfg, st):
    """Brief §Checks (MAX_CLIPS, default 5 clips): the baseline path reproduces a hook-free
    UMTClassifier, a dose=0 edit matches it (allclose 1e-5, max diff printed), worst mass
    gap from arms.csv."""
    from umt_mlm_feature_contrast import clip_loader
    from umt_wrapper import UMTClassifier
    from ToT_utils import CHECKPOINT_REGISTRY
    rows = clip_table(cfg).head(cfg["max_clips"] or 5)
    ref = UMTClassifier.from_pretrained(CHECKPOINT_REGISTRY[("umt", "ssv2")]).to(cfg["device"]).eval()
    D = st["sae"].get_dictionary()
    res = [check_clip(cfg, st, ref, D, rows, i, pv.to(cfg["device"]))
           for pv, i in clip_loader(rows, cfg, "R")]
    del ref
    print(f"check ok: worst baseline |diff| {max(d for d, _ in res):.2e}; "
          f"worst dose=0 diff {max(d0 for _, d0 in res):.2e} (< 1e-5 asserted); worst mass gap "
          f"{pd.read_csv(cfg['out_dir'] / 'arms.csv').mass_gap.max():.3f}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as both reference scripts
    torch.backends.cudnn.allow_tf32 = False
    mode = os.environ.get("MODE", sys.argv[1] if len(sys.argv) > 1 else "all")
    cfg = dict(CFG)
    if cfg["max_clips"]:
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"mode={mode} device={cfg['device']} out={cfg['out_dir']} conds={cfg['conds']} "
          f"dose={cfg['dose']} max_clips={cfg['max_clips'] or 'all'}")
    st = setup(cfg) if mode in ("mass", "ablate", "check", "all") else None
    if mode in ("mass", "all"):
        run_mass(cfg, st)
    if mode in ("check", "all"):
        run_check(cfg, st)
    if mode in ("ablate", "all"):
        run_ablate(cfg, st)
    if mode in ("summarise", "all"):
        run_summarise(cfg)
    if mode == "compare":
        run_compare(cfg)


if __name__ == "__main__":
    main()
