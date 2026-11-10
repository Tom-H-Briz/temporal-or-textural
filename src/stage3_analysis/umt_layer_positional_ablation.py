"""UMT positional-feature ablation + per-clip frame profiles at L5/L7/L9 (brief 051126).

Part 2: ablate the selected arms (ramp / nonramp / positional / near_flat / all, from
umt_layer_positional_select.py's sets_L{L}.csv) on all clips of the 35 SL classes under the
VTC 174-way head, SPLICE=1 by default, storing all 174 logits per clip x arm (fp16). No
controls (user, 5 Nov): the arms are compared with each other. Part 3: for R, C and
Rev, store each clip's (12, 6144) frame profile (z0 summed over patches) and re-run the
selection rules on the post-fix profiles + the near-flat always-on vs content-timed check.
Part 4: per-class summaries, temporal-vs-static gaps, and the cross-layer table.
Everything else is imported from umt_vtc_clock_ablation (clip_table, setup, full_logits,
clip_pass, edited_stream, arm_logits, clip_rows, measures, the table functions).
Usage: MODE=mass LAYER=5 uv run python src/stage3_analysis/umt_layer_positional_ablation.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]

import umt_vtc_clock_ablation as clock

MODEL = os.environ.get("MODEL", "umt")                # umt | videomae | timesformer (same test, 6 Nov)
TAG = {"umt": "umt", "videomae": "vm", "timesformer": "tf"}[MODEL]   # output-dir prefix

CFG = {
    "layer": int(os.environ.get("LAYER", 9)), "sae_k": 64, "job_label": "7ep",
    "model": MODEL, "dose": 1.0,
    "splice": os.environ.get("SPLICE", "1") == "1",
    "conds": os.environ.get("CONDS", "R"),
    # VM: C1 = tubelet-pair shuffle; Rev1 = tubelet order reversed (in-grammar); Rev kept for comparison
    "profile_conds": ("R", "C1", "Rev", "Rev1") if MODEL == "videomae" else ("R", "C", "Rev"),   # TF: as UMT
    "sets_csv": ROOT / f"outputs/analysis/{TAG}_layer_positional/sets_L{{L}}.csv",
    "xlayer_dir": ROOT / f"outputs/analysis/{TAG}_layer_positional",
    "sl_csv": ROOT / "outputs/Laura_SL/Static_Temporal.csv",
    "manifest": ROOT / "outputs/manifests/manifest_ssv2_all174_cap40.json",
    "labels_path": os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "video_dir": os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "out_dir": Path(os.environ.get("OUT_DIR", str(ROOT / f"outputs/analysis/{TAG}_layer_positional_ablation_L{{L}}"))),
    "max_clips": int(os.environ.get("MAX_CLIPS", 0)), "edit_batch": 16,
    "num_workers": int(os.environ.get("NUM_WORKERS", 4)),
    "n_boot": 1000, "seed": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def layer_path(cfg, key):
    """CFG path with {L} substituted by the layer index."""
    return Path(str(cfg[key]).format(L=cfg["layer"]))


def loader(cfg, st, rows, cond):
    """Batch-1 (pv, row-index) loader. UMT: umt_mlm_feature_contrast's (R / C). VM / TF:
    umt_vision_pair_contrast's (R / C1 / C) with row indices as labels, so done-tracking still works."""
    if cfg["model"] == "umt":
        from umt_mlm_feature_contrast import clip_loader
        return clip_loader(rows, cfg, "C" if cond == "C" else "R")
    from umt_vision_pair_contrast import clip_loader
    return clip_loader(list(zip(rows.clip_id, rows.index)), cfg, cond if cond in ("C", "C1") else "R",
                       st["processor"])                # VM C1 / TF C; Rev(1) applied after loading


def condition(pv, cond):
    """Rev = whole time axis flipped (apply_condition). Rev1 (VM) = the 2-frame tubelets in
    reverse order, frames inside each tubelet kept in order (the C1 analogue)."""
    if cond == "Rev1":
        B, T = pv.shape[:2]
        return pv.reshape(B, T // 2, 2, *pv.shape[2:]).flip(1).reshape(pv.shape)
    from umt_vision_pair_contrast import apply_condition
    return apply_condition(pv, cond)


def arm_sets(cfg):
    """sets_L{L}.csv -> the six ablated arms: the four selection arms (ramp, nonramp, near_flat,
    content_timed — per-clip rule 3, 6 Nov) plus the positional (ramp + nonramp) and all
    (every selected feature) unions. Nothing hardcoded per layer."""
    sel = pd.read_csv(str(layer_path(cfg, "sets_csv")))
    sel = sel[sel.arm.notna() & (sel.arm.astype(str) != "")]
    f = {a: sorted(g.feature.astype(int).tolist()) for a, g in sel.groupby("arm")}
    g = {a: f.get(a, []) for a in ("ramp", "nonramp", "near_flat", "content_timed")}
    return {"ramp": g["ramp"], "nonramp": g["nonramp"], "positional": g["ramp"] + g["nonramp"],
            "near_flat": g["near_flat"], "content_timed": g["content_timed"], "all": sum(g.values(), [])}


def arm_rows(cfg):
    """[(arm, feats)] in fixed order: baseline, batch_base (the reference deltas measure
    against), then the five arms. No control arms (user, 5 Nov)."""
    return [("baseline", []), ("batch_base", [])] + list(arm_sets(cfg).items())


def mass_rows(cfg, mass):
    """One row per arm: features, mean mass and share of the total activation mass (the
    cross-layer table needs it next to the deltas). No controls (user, 5 Nov: the arms are
    each other's comparison); mass_gap stays NaN only because clock.run_check prints it."""
    return [dict(arm=name, features=",".join(map(str, fs)), k=len(fs),
                 mass=float(mass[fs].mean()), mass_share=float(mass[fs].sum() / mass.sum()),
                 mass_gap=np.nan)
            for name, fs in arm_sets(cfg).items()]


def write_arms(cfg, mass):
    """Mass-stage outputs: mass_R.npy + arms.csv (one row per arm, no controls)."""
    g = pd.DataFrame(mass_rows(cfg, mass))
    g.to_csv(cfg["out_dir"] / "arms.csv", index=False)
    np.save(cfg["out_dir"] / "mass_R.npy", mass)
    print(g[["arm", "k", "mass", "mass_share"]].to_string(index=False))


def run_mass(cfg, st):
    """Stage 1 (GPU): mean per-feature SAE activation mass over the SL clips (R), then the
    per-arm mass share. One forward per clip; the loader's skip-to-next is detectable by index."""
    rows, total, n, done, t0 = clock.clip_table(cfg), None, 0, set(), time.time()
    for pv, i in loader(cfg, st, rows, "R"):
        if int(i) in done:
            continue
        done.add(int(i))
        _, _, z0 = clock.clip_pass(st, pv.to(cfg["device"]))
        s = z0.sum(0).cpu().numpy()
        total = s if total is None else total + s
        n += 1
        if n % 200 == 0:
            print(f"  mass {n}/{len(rows)} clips, {(time.time() - t0) / n:.2f} s/clip", flush=True)
    write_arms(cfg, total / max(n, 1))
    print(f"mass stage: {len(done)}/{len(rows)} clips in {(time.time() - t0) / 60:.1f} min")


def run_ablate(cfg, st):
    """Stage 2 (GPU): baseline + every arm per clip x condition -> per_clip_logits.parquet
    (174 fp16 logits per row, base_correct / ref_correct stored), rewritten after each
    condition. Identical to the clock driver's but with this brief's arm_rows."""
    arms = arm_rows(cfg)
    edits = [(a, f) for a, f in arms if a != "baseline"]
    D = st["sae"].get_dictionary()
    out = []
    for cond in cfg["conds"].split("+"):
        clock.ablate_cond(cfg, st, arms, edits, D, cond, out, lambda r, c: loader(cfg, st, r, c))
        df = pd.DataFrame(out)
        df[clock.LG] = df[clock.LG].astype(np.float16)
        df.to_parquet(cfg["out_dir"] / "per_clip_logits.parquet", index=False)
    print(f"ablate: {len(out)} clip-arm rows -> per_clip_logits.parquet")


def check_reshape(z0, prof):
    """Brief §Checks: the (12, 196, 6144) reshape must not move mass — the profile summed
    over frames equals z0 summed over tokens."""
    lhs, rhs = prof.double().sum(0), z0.double().sum(0)
    rel = float((lhs - rhs).abs().max() / rhs.abs().max())
    assert rel < 1e-4, f"frame reshape check failed: relative error {rel:.2e}"
    print(f"frame reshape check ok (max rel err {rel:.2e})")


def run_profiles(cfg, st):
    """Part 3 (GPU): per condition, z0 from clip_pass -> gather_by_position(..., 'umt') ->
    (12, 196, 6144) -> sum over patches -> one fp16 (12, 6144) profile per clip, saved as
    profiles_{cond}.npz (clip_id, class_id, prof). C and Rev are encode-pass only."""
    from ToT_utils import gather_by_position
    rows = clock.clip_table(cfg)
    for cond in cfg["profile_conds"]:
        profs, ids, cls, done, t0 = [], [], [], set(), time.time()
        for pv, i in loader(cfg, st, rows, cond):
            if int(i) in done:
                continue
            done.add(int(i))
            _, _, z0 = clock.clip_pass(st, condition(pv, cond).to(cfg["device"]))
            prof = gather_by_position(z0, cfg["model"]).sum(1)
            if not profs:
                check_reshape(z0, prof)
            profs.append(prof.half().cpu().numpy())
            r = rows.iloc[int(i)]
            ids.append(r.clip_id)
            cls.append(int(r.class_id))
        np.savez_compressed(cfg["out_dir"] / f"profiles_{cond}.npz",
                            clip_id=np.array(ids), class_id=np.array(cls, dtype=np.int64),
                            prof=np.stack(profs).astype(np.float16))
        print(f"profiles {cond}: {len(ids)} clips in {(time.time() - t0) / 60:.1f} min", flush=True)
    if cfg["model"] == "umt":                         # UMT-only, superseded checks
        run_recheck(cfg)
        run_nearflat(cfg)


def load_profiles(cfg, cond):
    """profiles_{cond}.npz -> (clip_id, class_id, prof (n, 12, 6144) float32)."""
    with np.load(str(cfg["out_dir"] / f"profiles_{cond}.npz")) as z:
        return z["clip_id"], z["class_id"], z["prof"].astype(np.float32)


def class_mean_profiles(prof, cls):
    """(n, 12, k) clip profiles -> (n_classes, 12, k) class means (post-fix re-check input)."""
    return np.stack([prof[cls == c].mean(0) for c in np.unique(cls)]).transpose(0, 2, 1)


def recheck_rules(R, C):
    """Rules 3-4 on post-fix class-mean profiles (R, C: (n_classes, k, 12)): per feature,
    r_RC = median over classes, cv and ramp_r from the class-mean R profile, and the arm
    under the same cuts as the selection."""
    from umt_layer_positional_select import CFG as SEL, pearson_rows
    r_rc = np.median(pearson_rows(R, C), axis=0)
    prof = R.mean(0)
    cv = prof.std(1, ddof=1) / prof.mean(1)
    ramp_r = pearson_rows(prof, np.broadcast_to(np.arange(12.0), prof.shape))
    arm = np.where(cv < SEL["cv_cut"], "near_flat",
                   np.where(np.abs(ramp_r) >= SEL["ramp_cut"], "ramp", "nonramp"))
    arm = np.where(r_rc >= SEL["r_min"], arm, "")
    return pd.DataFrame(dict(feature=np.arange(R.shape[1]), r_RC_post=r_rc,
                             cv_post=cv, ramp_r_post=ramp_r, arm_post=arm))


def run_recheck(cfg):
    """Part 3 CPU analysis 1: re-run rules 3-4 on the post-fix profiles (class means over
    the SL clips) and report every selected feature whose arm would change. Nothing is
    silently changed — the full comparison goes to refix_check_L{L}.csv and stdout."""
    sel = pd.read_csv(str(layer_path(cfg, "sets_csv")))
    sel = sel[sel.arm.notna() & (sel.arm.astype(str) != "")][["feature", "arm"]]
    sel = sel.rename(columns={"arm": "arm_pre"})
    _, cls, pr = load_profiles(cfg, "R")
    _, _, pc = load_profiles(cfg, "C")
    post = recheck_rules(class_mean_profiles(pr, cls), class_mean_profiles(pc, cls))
    out = sel.merge(post, on="feature")
    out.to_csv(cfg["out_dir"] / f"refix_check_L{cfg['layer']}.csv", index=False)
    changed = out[out.arm_pre != out.arm_post]
    print(f"refix check L{cfg['layer']}: {len(changed)} of {len(out)} selected features change arm")
    if len(changed):
        print(changed.to_string(index=False))


def feature_clip_stats(pr, rev):
    """Per-clip diagnostics for one feature's (n, 12) R profiles and (n, 12) Rev profiles:
    clip CV (ddof 1), peak-frame spread, does the Rev peak follow the content (11 - R peak)
    or stay at its position, and Rev-vs-flipped-R / Rev-vs-R profile correlations."""
    from umt_layer_positional_select import pearson_rows
    cv = pr.std(1, ddof=1) / np.maximum(pr.mean(1), 1e-12)
    peak_r, peak_v = pr.argmax(1), rev.argmax(1)
    r_flip = pearson_rows(rev, pr[:, ::-1])
    r_same = pearson_rows(rev, pr)
    return dict(cv_clip_med=float(np.median(cv)), cv_clip_frac_ge_02=float((cv >= 0.2).mean()),
                peak_n_unique=int(np.unique(peak_r).size), peak_mode_frac=float(
                    np.bincount(peak_r, minlength=12).max() / len(peak_r)),
                peak_std=float(peak_r.std()),
                rev_follow_frac=float((peak_v == 11 - peak_r).mean()),
                rev_stay_frac=float((peak_v == peak_r).mean()),
                rev_r_flip=float(np.nanmean(r_flip)), rev_r_same=float(np.nanmean(r_same)))


def verdict(s):
    """Always on = flat in each clip. Content-timed = peaked in each clip (median clip CV
    >= 0.2), the peak frame moves (more than one distinct peak), and it follows content
    under Rev (follow beats stay, and the Rev profile matches the flipped R profile better).
    Anything else is reported as ambiguous."""
    peaked = s["cv_clip_med"] >= 0.2
    moves = s["peak_n_unique"] > 1
    follows = s["rev_follow_frac"] > s["rev_stay_frac"] and s["rev_r_flip"] > s["rev_r_same"]
    return "always_on" if not peaked else ("content_timed" if moves and follows else "ambiguous")


def run_nearflat(cfg):
    """Part 3 CPU analysis 2: for every selected feature (near_flat first, positional for
    comparison), the per-clip diagnostics of feature_clip_stats under R and Rev -> verdict.
    Writes nearflat_L{L}.csv."""
    sel = pd.read_csv(str(layer_path(cfg, "sets_csv")))
    sel = sel[sel.arm.notna() & (sel.arm.astype(str) != "")][["feature", "arm"]]
    _, _, pr_all = load_profiles(cfg, "R")
    _, _, pv_all = load_profiles(cfg, "Rev")
    rows = []
    for r in sel.itertuples():
        fs = int(r.feature)
        rows.append(dict(feature=fs, arm=r.arm,
                         **feature_clip_stats(pr_all[:, :, fs], pv_all[:, :, fs])))
    out = pd.DataFrame(rows)
    out["verdict"] = [verdict({k: v for k, v in row.items() if k not in ("feature", "arm")})
                      for _, row in out.iterrows()]
    out.to_csv(cfg["out_dir"] / f"nearflat_L{cfg['layer']}.csv", index=False)
    print(f"near-flat vs positional verdicts L{cfg['layer']}:")
    print(out.groupby(["arm", "verdict"]).size().to_string())


def run_summarise(cfg):
    """Part 4 (CPU): the clock driver's summarise with this brief's arms — per-class d logit
    and d margin + CIs, a separate accuracy table, temporal-vs-static gaps, lean, then the
    cross-layer table across whatever layers have finished."""
    cl, L, arms = clock.load_logits(cfg)
    y = cl.class_id.to_numpy()
    meas = clock.measures(L, y, arms.index("batch_base"))
    ct = clock.class_table(meas, y, arms, cl, cfg)
    clock.acc_table(meas, y, arms, cl, cfg)
    at = pd.read_csv(cfg["out_dir"] / "summary_by_class_accuracy.csv")
    pd.concat([clock.sl_gap(ct, v, cfg["n_boot"], cfg["seed"]) for v in ("d_logit", "d_margin")],
              ignore_index=True).to_csv(cfg["out_dir"] / "summary_sl.csv", index=False)
    clock.sl_gap(at, "d_acc", cfg["n_boot"], cfg["seed"]).to_csv(
        cfg["out_dir"] / "summary_sl_accuracy.csv", index=False)
    clock.lean_table(L, arms).to_csv(cfg["out_dir"] / "lean.csv", index=False)
    print(ct.groupby("arm")[["d_logit", "d_margin"]].mean().round(3).to_string())
    print("summarise done: summary_by_class(.accuracy).csv, summary_sl(.accuracy).csv, lean.csv")


def layer_result(cfg, layer):
    """One finished layer -> rows of the cross-layer table: d logit / d margin / d accuracy
    per arm vs batch_base (no controls), next to the arm's share of the total activation
    mass (arms.csv)."""
    d = ROOT / "outputs/analysis" / f"{TAG}_layer_positional_ablation_L{layer}{'_splice' if cfg['splice'] else ''}"
    if not (d / "per_clip_logits.parquet").exists():
        return []
    cl, L, arms = clock.load_logits({**cfg, "out_dir": d})
    meas = clock.measures(L, cl.class_id.to_numpy(), arms.index("batch_base"))
    info = pd.read_csv(d / "arms.csv").set_index("arm")
    return [dict(layer=layer, arm=g, d_logit=float(meas[0][:, arms.index(g)].mean()),
                 d_margin=float(meas[1][:, arms.index(g)].mean()),
                 d_acc=float(meas[2][:, arms.index(g)].mean()),
                 mass_share=float(info.loc[g, "mass_share"]))
            for g in ("ramp", "nonramp", "positional", "near_flat", "content_timed", "all")]


def xlayer_table(cfg):
    """The brief's main result: d margin / d accuracy for every arm at
    L5 / L7 / L9, each with its share of the total activation mass removed. Reads the full
    (non-smoke) splice dirs of every layer that has run; writes xlayer_summary.csv."""
    rows = [r for layer in (5, 7, 9) for r in layer_result(cfg, layer)]
    tab = pd.DataFrame(rows)
    out = cfg["xlayer_dir"] / f"xlayer_summary{'' if cfg['splice'] else '_raw'}.csv"   # raw-stream check: own table
    tab.to_csv(out, index=False)
    print(f"cross-layer table -> {out}")
    if len(tab):
        print(tab.round(4).to_string(index=False))


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    mode = os.environ.get("MODE", sys.argv[1] if len(sys.argv) > 1 else "mass")
    cfg = dict(CFG)
    if cfg["splice"]:
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name.format(L=cfg["layer"]) + "_splice")
    else:
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name.format(L=cfg["layer"]))
    if cfg["max_clips"]:
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
        cfg["sets_csv"] = Path(str(cfg["sets_csv"]).replace(".csv", "_smoke.csv"))   # select's smoke sets
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"mode={mode} layer=L{cfg['layer']} device={cfg['device']} out={cfg['out_dir']} "
          f"conds={cfg['conds']} splice={cfg['splice']} max_clips={cfg['max_clips'] or 'all'}")
    st = clock.setup(cfg) if mode in ("mass", "check", "ablate", "profiles") else None
    if mode == "mass":
        run_mass(cfg, st)
    if mode == "arms":                                # CPU: re-derive arms.csv for new sets from mass_R.npy
        write_arms(cfg, np.load(cfg["out_dir"] / "mass_R.npy"))
    if mode == "check":
        clock.run_check(cfg, st)
    if mode == "ablate":
        run_ablate(cfg, st)
    if mode == "profiles":
        run_profiles(cfg, st)
    if mode == "summarise":
        run_summarise(cfg)
        xlayer_table(cfg)


if __name__ == "__main__":
    main()
