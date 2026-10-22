"""UMT MLM group ablation (brief mlm_group_ablation_brief_201026.md): ablate three
groups of 5 L9 vision SAE features per clean pair (gated / flip / no change, defined
on the contrast run's selection table), measure the damage to the refit MLM head's
masked pair-word T_ab and to every other word-piece in the caption, on held-out (g34)
clips under R and Rev, against mass-matched random control sets and single features,
with first-order predicted changes from one baseline gradient pass.

Modes (uv run python src/stage3_analysis/umt_mlm_group_ablation.py <mode>):
  groups     CPU: group membership + mass-matched controls -> groups.csv (needs the
             contrast run's sparse/ files for the masses)
  check      CPU, 2 clips, needs groups.csv: 7.2-7.4 — empty-edit identity and
             dose/residual are asserted; the baseline-vs-contrast-run diff is
             reported only (odd-width webms decode nondeterministically, so the
             stored per_clip.parquet values are one draw of that flicker)
  ablate     GPU/SLURM: per_clip_edits.parquet + per_clip_other_words.parquet
  summarise  CPU: summary.csv (gap measures, bootstrap CIs) + summary_accuracy.csv
  hand       CPU: hand_overlap.csv — hand features among the group members,
             all five pairs (needs groups.csv here + the clean run's folders)
Push/pull (brief mlm_pushpull_brief_201026.md): PAIRS="93_86;94_87" CLIP_SOURCE=val
CONTRAST_DIR=.../umt_mlm_feature_contrast_pushpull OUT_DIR=..._pushpull <mode>
Usage: MODE=groups uv run python src/stage3_analysis/umt_mlm_group_ablation.py
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
from umt_mlm_feature_contrast import (clip_loader, clip_pass, clip_texts, feature_vecs,
                                      pair_rows, setup as contrast_setup)
from umt_mlm_refit import encode, mlm_fusion, mlm_text
from umt_bridge_phase3 import apply_edit, finish_tower, mass_controls, tower_to_layer
from umt_vision_pair_contrast import apply_condition

CFG = {
    "layer": 9, "sae_k": 64, "job_label": "7ep", "dose": 1.0,
    "pairs": {(45, 43): ("up", "down"), (46, 5): ("opening", "closing"),
              (168, 165): ("upwards", "downwards"),   # the three clean pairs
              (93, 86): ("pushing", "pulling"), (94, 87): ("pushing", "pulling")},
    "pairs_env":       os.environ.get("PAIRS", ""),    # subset, e.g. "93_86;94_87"
    "clip_source":     os.environ.get("CLIP_SOURCE", "phase3"),  # phase3 | val (§2 split)
    "labels_path":     os.environ.get("LABELS_PATH", str(ROOT / "data/ssv2/labels/labels.json")),
    "validation_path": os.environ.get("VALIDATION_PATH", str(ROOT / "data/ssv2/labels/validation.json")),
    "video_dir":       os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2_val_set")),
    "max_clips":       int(os.environ.get("MAX_CLIPS", 0)),   # local CPU checks only
    "contrast_dir": Path(os.environ.get("CONTRAST_DIR",
                          str(ROOT / "outputs/analysis/umt_mlm_feature_contrast"))),
    "phase3_dir": ROOT / "outputs/analysis/umt_vision_text_bridge/phase3",
    "refit_dir": ROOT / "outputs/analysis/umt_mlm_refit",
    "out_dir":     Path(os.environ.get("OUT_DIR",
                        str(ROOT / "outputs/analysis/umt_mlm_group_ablation"))),
    "clean_run":  (ROOT / "outputs/analysis/umt_mlm_feature_contrast",
                   ROOT / "outputs/analysis/umt_mlm_group_ablation"),  # hand check, §3 D4
    "hand_csv":   ROOT / "outputs/analysis/hand_tracker_alignment/umt_L9.csv",
    "device":      "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 4, "n_boot": 1000, "seed": 0,
    # SPLICE=1: DFAEngine.run_ablated convention — L9 replaced by decode(z, feats zeroed) + dim_mean
    # (no error term); baseline = the unablated splice. SPLICE=0: raw-stream subtraction (20 Oct run).
    "splice": os.environ.get("SPLICE", "0") == "1",
}

EXPECTED = {  # brief §2 membership table (acceptance 7.1)
    (45, 43): dict(gated=[4250, 393, 2181, 2736, 404], flip=[1063, 4474, 4265, 3948, 3583],
                   nochange=[541, 4606, 1694, 5883, 683]),
    (46, 5): dict(gated=[1789, 404, 2343, 1820, 1874], flip=[5473, 1694, 683, 6127, 968],
                  nochange=[6059, 2417, 4301, 3583, 2383]),
    (168, 165): dict(gated=[1684, 3884, 4525, 2472, 4144], flip=[2261, 541, 5473, 4788, 3419],
                     nochange=[113, 2483, 1694, 1063, 1768]),
}
GROUPS = ("gated", "flip", "nochange")


def group_table(tab):
    """§2 group rules on one pair's selection table (one row per feature): importance
    rank over active features, share of the pair total, size ratio, same-sign flag."""
    t = tab[(tab.act_mass_a + tab.act_mass_b) > 0].copy()
    t["importance"] = (t.mean_a.abs() + t.mean_b.abs()) / 2
    t["share"] = t.importance / t.importance.sum()
    t["size_ratio"] = np.minimum(t.mean_a.abs(), t.mean_b.abs()) / \
        np.maximum(t.mean_a.abs(), t.mean_b.abs())
    t["same_sign"] = np.sign(t.mean_a) == np.sign(t.mean_b)
    return t.sort_values("importance", ascending=False, kind="stable")


def pick_groups(t):
    """{group: [5 features, importance-desc]} under the §2 rules."""
    return {
        "gated": t[(t.act_ratio < 0.25) & t.same_sign].head(5).feature.tolist(),
        "flip": t[(t.act_ratio >= 0.9) & ~t.same_sign].head(5).feature.tolist(),
        "nochange": t[(t.act_ratio >= 0.9) & t.same_sign &
                      (t.size_ratio >= 0.9)].head(5).feature.tolist(),
    }


def pair_mass(cfg, clips):
    """(6144,) mean per-clip activation mass over one pair's R selection clips, from the
    contrast run's sparse files (feature_vecs' z-mass column)."""
    M = np.stack([feature_vecs(cfg["contrast_dir"] / "sparse", "R", c)[1] for c in clips])
    return M.mean(0)


def group_controls(t, mass, members):
    """One mass-matched control per member (§2): pool = active features in none of the
    pair's three groups and not already used as a control, matched as mass_controls
    does (nearest mass, heaviest member first, without replacement) -> DataFrame."""
    used = [f for g in GROUPS for f in members[g]]
    pool = set(t.feature.tolist()) - set(used)
    rows = []
    for g in GROUPS:
        sel = np.array(members[g])
        ctrl, gap = mass_controls(mass, sel, None, None, pool=pool)
        pool -= set(ctrl.tolist())
        rows += [dict(group=g, rank=i + 1, feature=int(f), control=int(c), mass=float(mass[f]),
                      control_mass=float(mass[c]), mass_gap=float(gp))
                 for i, (f, c, gp) in enumerate(zip(sel, ctrl, gap))]
    return pd.DataFrame(rows)


def pair_accuracy(per):
    """§6.1: per pair x condition accuracy (sign of T_ab) and mean T_ab per class
    (orientation, §7.4: report, don't fix)."""
    rows = []
    for (a, b, cond), g in per.groupby(["a", "b", "condition"]):
        is_a = g.true_class == a
        rows.append(dict(a=a, b=b, condition=cond, n=len(g),
                         acc=float(((g.T_ab > 0) == is_a).mean()),
                         mean_T_ab_a=float(g[is_a].T_ab.mean()),
                         mean_T_ab_b=float(g[~is_a].T_ab.mean())))
    return pd.DataFrame(rows)


def print_contrast_summary(cfg, per):
    """§6.1 checkpoint: split.csv counts, mask skips, R/C/Rev accuracy + orientation."""
    split = cfg["contrast_dir"] / "split.csv"
    if split.exists():
        print("split counts:\n" + pd.read_csv(split, dtype={"clip_id": str})
              .groupby(["a", "b", "split"]).size().rename("n").to_string())
    sk = cfg["contrast_dir"] / "skipped_clips.csv"
    if sk.exists():
        df = pd.read_csv(sk)
        if len(df):
            print("mask skips:\n" + df.groupby(["a", "b", "reason"]).size().rename("n").to_string())
    print("pair accuracy / orientation:\n" +
          pair_accuracy(pd.read_parquet(cfg["contrast_dir"] / "per_clip.parquet"))
          .round(3).to_string(index=False))


def run_groups(cfg):
    """Stage 1 (CPU): group membership (asserted against the §2 table for the clean
    pairs; printed for review for new pairs) + mass-matched controls -> groups.csv."""
    print_contrast_summary(cfg, None)
    tab = pd.read_parquet(cfg["contrast_dir"] / "selection_table.parquet")
    per = pd.read_parquet(cfg["contrast_dir"] / "per_clip.parquet")
    out = []
    for (a, b), g in tab.groupby(["a", "b"]):
        if (a, b) not in cfg["pairs"]:
            continue
        t = group_table(g)
        members = pick_groups(t)
        for grp, feats in members.items():
            if (a, b) in EXPECTED:
                assert feats == EXPECTED[(a, b)][grp], \
                    f"{a}/{b} {grp}: {feats} != {EXPECTED[(a, b)][grp]}"
            elif len(feats) < 5:
                print(f"{a}/{b} {grp}: only {len(feats)} qualifying features (< 5)")
        clips = per[(per.a == a) & (per.b == b) & (per.condition == "R") &
                    (per.split == "selection")].clip_id
        ctrls = group_controls(t, pair_mass(cfg, clips), members)
        ti = t.set_index("feature")
        info = ctrls.join(ti[["share", "act_ratio", "size_ratio", "evidence", "bias"]]
                          .loc[ctrls.feature].reset_index(drop=True))
        info.insert(0, "a", a); info.insert(1, "b", b)
        info["opposer"] = info.evidence < 0
        out.append(info)
        print(f"{a}/{b} groups ok: " + "; ".join(
            f"{g}={[f for f in members[g]]}" for g in GROUPS))
        if (a, b) not in EXPECTED:   # new pairs: full table for §6.1 review
            cols = ["group", "rank", "feature", "share", "act_ratio", "size_ratio",
                    "evidence", "bias", "opposer", "control", "mass", "control_mass", "mass_gap"]
            print(info[cols].round(4).to_string(index=False))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(cfg["out_dir"] / "groups.csv", index=False)
    print(res.groupby(["a", "b", "group"]).mass_gap.max().to_string())
    print(f"groups.csv: {len(res)} rows (3 pairs x 3 groups x 5)")


def load_validation(cfg):
    """{clip_id: (cleaned label, placeholders)} from the validation json."""
    rows = json.load(open(cfg["validation_path"]))
    return {str(r["id"]): (r["label"], r["placeholders"]) for r in rows}


def caption_variants(cfg, s, cid, pair_text, val):
    """All masked variants of one caption as one batch: row 0 = the contrast run's
    pair-word mask, rows 1.. = every other word-piece masked alone (pair word visible).
    -> dict with ids/attn (device), masked positions, true tokens, is_obj (others)."""
    label, ph = val[cid]
    enc, _ = encode(s["tok"], s["pre_text"](label), 25)
    ids_pair, _, pos, id_a, id_b = pair_text
    assert enc[pos] in (id_a, id_b), f"{cid}: pair word moved"
    obj = set(s["tok"](s["pre_text"](" ".join(ph)), add_special_tokens=False)["input_ids"])
    others = [p for p in range(1, len(enc)) if p != pos]
    mask = s["tok"].mask_token_id
    rows_ids = [ids_pair[0].tolist()] + [[*enc[:p], mask, *enc[p + 1:]] for p in others]
    ids = torch.tensor(rows_ids, device=cfg["device"])
    return dict(ids=ids, attn=torch.ones_like(ids), pos=[pos] + others,
                true=[enc[p] for p in [pos] + others],
                is_obj=[enc[p] in obj for p in others],
                id_a=id_a, id_b=id_b, pair_text=pair_text)


def pair_edits(groups):
    """One pair's edit list from its groups.csv rows: [(name, kind, group, feats, k)] —
    3 groups of 5, 3 control sets, 15 singles (rank order within each) — plus the
    {feature: opposer} map for the singles' rows."""
    out = []
    for g in GROUPS:
        sub = groups[groups.group == g].sort_values("rank")
        feats = sub.feature.astype(int).tolist()
        out += [(g, "group", g, feats, len(feats)),
                (f"ctrl_{g}", "control", g, sub.control.astype(int).tolist(), len(feats))]
        out += [(f"single_{int(m.feature)}", "single", g, [int(m.feature)], 1)
                for _, m in sub.iterrows()]
    opp = dict(zip(groups.feature.astype(int), groups.opposer.astype(bool)))
    return out, opp


@torch.no_grad()
def edit_logits(s, h8, add, cap, x9, z0, D, feats, dose):
    """x' = x9 - sum(z_f d_f) (dose 1, ablation only) -> tower 10-11 -> fusion -> head
    at every masked position -> logits (n_variants, vocab)."""
    assert dose == 1.0, "ablation only: dose must be 1.0"
    if s.get("splice"):                               # DFAEngine._splice_hook: decode, add mean back
        z = z0.clone()
        z[:, list(feats)] = 0.0
        x = (s["sae"].decode(z) + s["dim_mean"]).unsqueeze(0).to(x9.dtype)
    else:
        x = apply_edit(x9, z0, D, feats, dose=dose)
    vis = finish_tower(s["model"], x)[0].unsqueeze(0)
    h = mlm_fusion(h8, add, vis, s["w"])
    sel = h[torch.arange(len(cap["pos"]), device=h.device),
            torch.tensor(cap["pos"], device=h.device)]
    return s["head"](sel)


def edit_measures(logits, base, cap, is_a, id_a, id_b):
    """One edit's per-clip measures (vs the baseline dict in `base`)."""
    lp = logits.log_softmax(-1).float()
    pair = lp[0]
    true_pair = id_a if is_a else id_b
    tr = torch.tensor(cap["true"][1:], dtype=torch.long, device=lp.device)
    lp_true = lp[1:].gather(1, tr[:, None]).squeeze(1)
    top1 = logits[1:].argmax(-1)
    is_obj = torch.tensor(cap["is_obj"], dtype=torch.bool, device=lp.device)
    if not cap["is_obj"]:
        return dict(T_ab=float(logits[0, id_a] - logits[0, id_b]),
                    logp_a=float(pair[id_a]), logp_b=float(pair[id_b]),
                    top1_id=int(logits[0].argmax()), d_logp_pair=float(
                        pair[true_pair] - base["lp_true_pair"]),
                    d_logp_other=np.nan, d_logp_other_obj=np.nan, d_logp_other_non=np.nan,
                    ow_logp=np.zeros(0, np.float32), ow_top1=np.zeros(0, np.int64))
    return dict(
        T_ab=float(logits[0, id_a] - logits[0, id_b]),
        logp_a=float(pair[id_a]), logp_b=float(pair[id_b]),
        top1_id=int(logits[0].argmax()),
        d_logp_pair=float(pair[true_pair] - base["lp_true_pair"]),
        d_logp_other=float((lp_true - base["lp_true_other"]).mean()),
        d_logp_other_obj=float((lp_true[is_obj] - base["lp_true_other"][is_obj]).mean()),
        d_logp_other_non=float((lp_true[~is_obj] - base["lp_true_other"][~is_obj]).mean()),
        ow_logp=lp_true.cpu().numpy(), ow_top1=top1.cpu().numpy())


def baseline_arrays(logits, cap, is_a, id_a, id_b):
    """The baseline dict edit_measures diffs against: pair-word log p(true) scalar,
    per-other-word log p(true) tensor, plus top-1 arrays for the other-words table."""
    lp = logits.log_softmax(-1).float()
    tr = torch.tensor(cap["true"][1:], dtype=torch.long, device=lp.device)
    lp_true = lp[1:].gather(1, tr[:, None]).squeeze(1)
    return dict(lp_true_pair=float(lp[0, id_a if is_a else id_b]),
                lp_true_other=lp_true,
                ow_top1=logits[1:].argmax(-1).cpu().numpy())


def clip_edits(cfg, s, r, cond, cap, h8, add, x9, z0, D, edits, opp):
    """All 22 passes for one clip-condition (baseline + 21 edits, §3): baseline measures,
    one baseline gradient pass for the first-order prediction, then every edit."""
    recs, owrecs = [], []
    base_logits = edit_logits(s, h8, add, cap, x9, z0, D, [], cfg["dose"])
    is_a, id_a = r.true_class == r.a, cap["id_a"]
    id_b = cap["id_b"]
    base = baseline_arrays(base_logits, cap, is_a, id_a, id_b)
    _, _, _, c = clip_pass(s, x9, cap["pair_text"])
    csum = c.sum(0)
    bm = edit_measures(base_logits, base, cap, is_a, id_a, id_b)
    all_edits = [("baseline", "baseline", "", [], 0)] + edits
    for name, kind, group, feats, k in all_edits:
        logits = base_logits if kind == "baseline" else \
            edit_logits(s, h8, add, cap, x9, z0, D, feats, cfg["dose"])
        m = edit_measures(logits, base, cap, is_a, id_a, id_b)
        recs.append(clip_edit_row(r, cond, name, kind, group, feats, k, m, bm,
                                  float(-csum[feats].sum()), cap,
                                  bool(opp.get(feats[0], False)) if kind == "single" else False))
        owrecs += other_word_rows(r, cond, name, kind, m, base, cap)
    return recs, owrecs


def clip_edit_row(r, cond, name, kind, group, feats, k, m, bm, pred, cap, opposer):
    """One per_clip_edits.parquet row (§4)."""
    return dict(clip_id=r.clip_id, a=r.a, b=r.b, true_class=r.true_class, split=r.split,
                condition=cond, edit=name, kind=kind, group=group, k=k, opposer=opposer,
                feats=",".join(map(str, feats)),
                T_ab_base=bm["T_ab"], T_ab=m["T_ab"], pred_dT=pred,
                logp_a=m["logp_a"], logp_b=m["logp_b"],
                logp_a_base=bm["logp_a"], logp_b_base=bm["logp_b"],
                top1_id=int(m["top1_id"]), top1_correct=int(m["top1_id"]) == cap["true"][0],
                d_logp_pair=m["d_logp_pair"], d_logp_other=m["d_logp_other"],
                d_logp_other_obj=m["d_logp_other_obj"], d_logp_other_non=m["d_logp_other_non"],
                n_other=len(cap["is_obj"]), n_obj=int(sum(cap["is_obj"])),
                n_nonobj=int(len(cap["is_obj"]) - sum(cap["is_obj"])))


def other_word_rows(r, cond, name, kind, m, base, cap):
    """per_clip_other_words.parquet rows for one edit (§4)."""
    return [dict(clip_id=r.clip_id, a=r.a, b=r.b, true_class=r.true_class, split=r.split,
                 condition=cond, edit=name, kind=kind, position=p, token=t, is_obj=bool(o),
                 logp_base=float(b), logp_edit=float(e),
                 top1_base=int(tb), top1_edit=int(te),
                 top1_correct_base=bool(tb == t), top1_correct_edit=bool(te == t))
            for p, t, o, b, e, tb, te in zip(cap["pos"][1:], cap["true"][1:], cap["is_obj"],
                                             base["lp_true_other"].tolist(), m["ow_logp"],
                                             base["ow_top1"], m["ow_top1"])]


def run_ablate(cfg, s):
    """Stage 2 (GPU): 22 edits x (R, Rev) for every g34 clip of the three pairs ->
    per_clip_edits.parquet + per_clip_other_words.parquet."""
    assert cfg["dose"] == 1.0, "ablation only"
    groups = pd.read_csv(cfg["out_dir"] / "groups.csv")
    edits = {k: pair_edits(g) for k, g in groups.groupby(["a", "b"])}
    rows = pair_rows(cfg)
    texts, skips = clip_texts(cfg, rows, s)
    keep = rows[rows.split == "g34"]
    keep = keep[keep.clip_id.isin(texts)].sort_values("clip_id").reset_index(drop=True)
    if cfg["max_clips"]:
        keep = keep.head(cfg["max_clips"]).reset_index(drop=True)
    val = load_validation(cfg)
    print(f"{len(rows)} g34 clips: {len(keep)} used, {len(skips)} mask-skipped")
    recs, owrecs, hcache, D, t0 = [], [], {}, s["sae"].get_dictionary(), time.time()
    for cond in ("R", "Rev"):
        filled = np.zeros(len(keep), bool)
        for pv, y in clip_loader(keep, cfg, cond):
            i = int(y)
            if filled[i]:
                continue
            r = keep.iloc[i]
            pv = apply_condition(pv, cond).to(cfg["device"])
            x9 = tower_to_layer(s["model"], pv, layer=cfg["layer"])
            with torch.no_grad():
                _, z0 = s["sae"].encode((x9[0] - s["dim_mean"]).float())
            if r.clip_id not in hcache:
                cap = caption_variants(cfg, s, r.clip_id, texts[r.clip_id], val)
                hcache[r.clip_id] = (cap, *mlm_text(cap["ids"], cap["attn"], s["w"]))
            cap, h8, add = hcache[r.clip_id]
            a, o = clip_edits(cfg, s, r, cond, cap, h8, add, x9, z0, D,
                              *edits[(r.a, r.b)])
            recs += a
            owrecs += o
            filled[i] = True
        done = int(filled.sum())
        print(f"  {cond}: {done}/{len(keep)} clips  "
              f"{(time.time() - t0) / max(len(recs) // 22, 1) * 1e3:.0f} ms/clip-cond", flush=True)
    pd.DataFrame(recs).to_parquet(cfg["out_dir"] / "per_clip_edits.parquet", index=False)
    pd.DataFrame(owrecs).to_parquet(cfg["out_dir"] / "per_clip_other_words.parquet", index=False)
    print(f"ablate done: {len(recs)} edit rows, {len(owrecs)} other-word rows, "
          f"{(time.time() - t0) / 60:.1f} min")


def run_check(cfg, s):
    """§6.2 checkpoint (CPU, 2 clips): acceptance 7.2-7.4."""
    groups = pd.read_csv(cfg["out_dir"] / "groups.csv")
    edits = {k: pair_edits(g) for k, g in groups.groupby(["a", "b"])}
    rows = pair_rows(cfg)
    texts, _ = clip_texts(cfg, rows, s)
    keep = rows[(rows.split == "g34") & rows.clip_id.isin(texts)] \
        .sort_values("clip_id").head(cfg["max_clips"] or 2).reset_index(drop=True)
    per = pd.read_parquet(cfg["contrast_dir"] / "per_clip.parquet")
    ref = per.set_index(["clip_id", "condition"]).T_ab
    val = load_validation(cfg)
    D = s["sae"].get_dictionary()
    for pv, y in clip_loader(keep, cfg, "R"):
        r = keep.iloc[int(y)]
        cid = r.clip_id
        x9 = tower_to_layer(s["model"], pv.to(cfg["device"]), layer=cfg["layer"])
        with torch.no_grad():
            _, z0 = s["sae"].encode((x9[0] - s["dim_mean"]).float())
        cap = caption_variants(cfg, s, cid, texts[cid], val)
        h8, add = mlm_text(cap["ids"], cap["attn"], s["w"])
        assert len(cap["pos"]) == len(cap["true"]) == len(cap["ids"]), "variant count"
        assert len(cap["is_obj"]) == cap["ids"].shape[1] - 2, "other-word count = len - 2"
        base = edit_logits(s, h8, add, cap, x9, z0, D, [], cfg["dose"])
        ident = edit_logits(s, h8, add, cap, apply_edit(x9, z0, D, [], dose=1.0),
                            z0, D, [], cfg["dose"])
        assert torch.equal(base, ident), "7.2: empty edit is not the identity"
        f = next(e[3][0] for e in edits[(r.a, r.b)][0]
                 if e[1] == "single" and e[2] == "gated")   # first gated single
        with torch.no_grad():
            resid = (x9 - apply_edit(x9, z0, D, [f], dose=1.0))[0] - (z0[:, [f]] @ D[[f]])
        assert resid.abs().max() < 1e-5, f"7.3: residual {resid.abs().max():.2e}"
        t_ab = float(base[0, cap["id_a"]] - base[0, cap["id_b"]])
        d = abs(t_ab - float(ref.loc[(cid, "R")]))
        odd = pv.shape[-1] % 2 == 1
        print(f"  {cid}: 7.2 ok, 7.3 ok (feature {f}), T_ab {t_ab:.4f} vs contrast "
              f"{float(ref.loc[(cid, 'R')]):.4f} |diff| {d:.2e}"
              + (" (odd width: decode flicker)" if odd else "") + " [7.4 report-only]")
    print("check ok: 7.2 identity, 7.3 dose/residual asserted; "
          "7.4 baseline-vs-contrast reported above (nondeterministic odd-width decode)")


def boot_stats(dT, is_a, pred, idx):
    """Point estimates for one (pair, condition, edit) clip set, and the same on
    bootstrap index draws idx (n_boot, n_clips): §4a gap quantities."""
    def stats(dT_, ia_, pr_):
        ma, mb = dT_[ia_].mean(), dT_[~ia_].mean()
        pa, pb = pr_[ia_].mean(), pr_[~ia_].mean()
        sep, bias = (ma - mb) / 2, (ma + mb) / 2
        psep, pbias = (pa - pb) / 2, (pa + pb) / 2
        div = lambda x, y: x / y if abs(y) > 1e-12 else np.nan
        return np.array([ma, mb, sep, bias, psep, pbias,
                         div(sep, psep), div(bias, pbias), dT_.mean()])
    point = stats(dT, is_a, pred)
    boots = np.stack([stats(dT[i], is_a[i], pred[i]) for i in idx])
    return point, boots


def resample_idx(n, n_boot, seed):
    """(n_boot, n) clip index draws, seed 0 per group (deterministic)."""
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


GAP_COLS = ["dT_a", "dT_b", "d_sep", "d_bias", "pred_sep", "pred_bias",
            "sep_over_pred", "bias_over_pred", "dT_all"]
OW_COLS = ["d_logp_pair", "d_logp_other", "d_logp_other_obj", "d_logp_other_non"]


def summary_row(g, splits, n_boot, seed):
    """One summary.csv row for a (pair, condition, edit) group of clips."""
    dT = g.dT.to_numpy()
    is_a = (g.true_class == g.a.iloc[0]).to_numpy()
    idx = resample_idx(len(g), n_boot, seed)
    point, boots = boot_stats(dT, is_a, g.pred_dT.to_numpy(), idx)
    pr = g.pred_dT.to_numpy()
    ok = np.abs(pr) > 0
    row = dict(a=g.a.iloc[0], b=g.b.iloc[0], condition=g.condition.iloc[0],
               edit=g.edit.iloc[0], kind=g.kind.iloc[0], group=g.group.iloc[0],
               n=len(g), n_a=int(is_a.sum()), n_b=int((~is_a).sum()),
               act_over_pred_med=float(np.nanmedian(dT[ok] / pr[ok])) if ok.any() else np.nan)
    for c, v, bt in zip(GAP_COLS, point, boots.T):
        row[c] = float(v)
        row[c + "_lo"], row[c + "_hi"] = np.nanpercentile(bt, [2.5, 97.5])
    for c in OW_COLS:
        vals = g[c].to_numpy()
        row[c] = float(vals.mean())
        lo, hi = np.nanpercentile(vals[idx].mean(1), [2.5, 97.5])
        row[c + "_lo"], row[c + "_hi"] = float(lo), float(hi)
    for nm, arr in splits.items():
        row[nm] = float(arr.mean())
        lo, hi = np.nanpercentile(arr[idx].mean(1), [2.5, 97.5])
        row[nm + "_lo"], row[nm + "_hi"] = float(lo), float(hi)
    return row


def opposer_singles(df, grp):
    """Per-clip helper/opposer sums of single-feature dT for one group's pair-condition:
    {(condition, clip_id): (helper_sum, opposer_sum)}."""
    singles = df[(df.kind == "single") & (df.group == grp)][
        ["clip_id", "condition", "dT", "opposer"]]
    out = {}
    for (cid, cond), g in singles.groupby(["clip_id", "condition"]):
        out[(cond, cid)] = (float(g[~g.opposer].dT.sum()), float(g[g.opposer].dT.sum()))
    return out


def accuracy_row(g, ow):
    """One summary_accuracy.csv row: pair-word sign accuracy + other-word top-1 (§4a)."""
    is_a = (g.true_class == g.a.iloc[0]).to_numpy()
    acc = ((g.T_ab.to_numpy() > 0) == is_a).mean()
    acc0 = ((g.T_ab_base.to_numpy() > 0) == is_a).mean()
    return dict(a=g.a.iloc[0], b=g.b.iloc[0], condition=g.condition.iloc[0],
                edit=g.edit.iloc[0], kind=g.kind.iloc[0], group=g.group.iloc[0],
                pair_acc_base=float(acc0), pair_acc=float(acc), pair_acc_change=float(acc - acc0),
                **ow)


def top1_stats(m):
    """Other-word top-1 base/edit/change over one clip x edit row set (§4a)."""
    out = {}
    for p, sub in (("all", m), ("obj", m[m.is_obj]), ("non", m[~m.is_obj])):
        e, b = float(sub.top1_correct_edit.mean()), float(sub.top1_correct_base.mean())
        out[f"other_top1_{p}_base"], out[f"other_top1_{p}_edit"] = b, e
        out[f"other_top1_{p}_change"] = e - b
    return out


def run_summarise(cfg):
    """Stage 3 (CPU): summary.csv (gap measures + CIs + opposer split) and
    summary_accuracy.csv (accuracy changes only)."""
    df = pd.read_parquet(cfg["out_dir"] / "per_clip_edits.parquet")
    ow = pd.read_parquet(cfg["out_dir"] / "per_clip_other_words.parquet")
    groups = pd.read_csv(cfg["out_dir"] / "groups.csv")
    df["dT"] = df.T_ab - df.T_ab_base
    opp = {(a, b, g): opposer_singles(df, g)
           for (a, b), _ in groups.groupby(["a", "b"]) for g in GROUPS}
    rows, accs = [], []
    for (a, b, cond, edit), g in df.groupby(["a", "b", "condition", "edit"]):
        splits = {}
        if g.kind.iloc[0] == "group":
            sums = opp[(a, b, g.group.iloc[0])]
            splits = {nm: np.array([sums[(cond, c)][i] for c in g.clip_id])
                      for i, nm in enumerate(("helper_sum_dT", "opposer_sum_dT"))}
        rows.append(summary_row(g, splits, cfg["n_boot"], cfg["seed"]))
        m = ow[(ow.condition == cond) & (ow.edit == edit) & ow.clip_id.isin(set(g.clip_id))]
        accs.append(accuracy_row(g, top1_stats(m)))
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "summary.csv", index=False)
    pd.DataFrame(accs).to_csv(cfg["out_dir"] / "summary_accuracy.csv", index=False)
    sing = df[df.kind == "single"]
    print(f"summary: {len(rows)} rows; predicted-vs-actual dT over singles: "
          f"r = {np.corrcoef(sing.pred_dT, sing.dT)[0, 1]:.3f} (n = {len(sing)})")


def enrichment_row(a, b, tab, is_hand, note):
    """§3 D4: hand fraction among the pair's top 1% by importance vs all active."""
    t = group_table(tab[(tab.a == a) & (tab.b == b)])
    n_top = int(np.ceil(0.01 * len(t)))   # same top-1% cut as the group analysis (62/62/60)
    f_top = float(t.feature.head(n_top).map(lambda f: bool(is_hand.loc[f])).mean())
    f_all = float(t.feature.map(lambda f: bool(is_hand.loc[f])).mean())
    return dict(row_type="enrichment", a=a, b=b, n_active=len(t), n_top1pct=n_top,
                top1pct_hand_frac=f_top, active_hand_frac=f_all,
                enrich_ratio=f_top / f_all if f_all > 0 else np.nan, note=note)


def hand_rows(groups, tab, hand, is_hand, note):
    """member / group / enrichment rows for one run's pairs (§3 D4)."""
    rows = []
    for (a, b), g in groups.groupby(["a", "b"]):
        for _, m in g.sort_values(["group", "rank"]).iterrows():
            h = hand.loc[m.feature]
            rows.append(dict(row_type="member", a=a, b=b, group=m.group, rank=int(m["rank"]),
                             feature=m.feature, share=m.share, overlap=h.overlap,
                             track=h.track, coact=h.coact,
                             is_hand=bool(is_hand.loc[m.feature]), note=note))
        for grp, sub in g.groupby("group"):
            nh = sub[sub.feature.map(lambda f: bool(is_hand.loc[f]))]
            rows.append(dict(row_type="group", a=a, b=b, group=grp, n_members=len(sub),
                             n_hand=len(nh),
                             hand_importance_share=float(nh.share.sum() / sub.share.sum()),
                             note=note))
        rows.append(enrichment_row(a, b, tab, is_hand, note))
    return rows


def run_hand(cfg):
    """§3 D4: hand-feature overlap of the group members, all five pairs -> hand_overlap.csv."""
    hand = pd.read_csv(cfg["hand_csv"]).set_index("feature")
    is_hand = (hand.coact >= 0.5) & (hand.overlap > 0.4)
    note = "hand alignment computed on move up/down clips"
    runs = [(cfg["contrast_dir"], cfg["out_dir"])]
    if (cfg["contrast_dir"], cfg["out_dir"]) != tuple(cfg["clean_run"]):
        runs.append((Path(cfg["clean_run"][0]), Path(cfg["clean_run"][1])))
    t = hand.loc[2782]
    rows = [dict(row_type="definition", n_hand_features=int(is_hand.sum()), note=note),
            dict(row_type="template", feature=2782, overlap=t.overlap, track=t.track,
                 coact=t.coact, is_hand=bool(is_hand.loc[2782]), note=note)]
    for contrast_dir, abl_dir in runs:
        rows += hand_rows(pd.read_csv(abl_dir / "groups.csv"),
                          pd.read_parquet(contrast_dir / "selection_table.parquet"),
                          hand, is_hand, note)
    pd.DataFrame(rows).to_csv(cfg["out_dir"] / "hand_overlap.csv", index=False)
    n_hand = sum(1 for r in rows if r["row_type"] == "member" and r["is_hand"])
    print(f"hand_overlap.csv: {len(rows)} rows; {n_hand} hand members; "
          f"{int(is_hand.sum())} hand features of {len(hand)}")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    torch.backends.cudnn.allow_tf32 = False
    mode = os.environ.get("MODE", sys.argv[1] if len(sys.argv) > 1 else "all")
    cfg = dict(CFG)
    if cfg["pairs_env"]:                              # subset, e.g. push/pull runs
        keep = {tuple(map(int, p.split("_"))) for p in cfg["pairs_env"].split(";")}
        cfg["pairs"] = {p: w for p, w in cfg["pairs"].items() if p in keep}
    if cfg["splice"]:                                 # keep raw-stream and splice results apart
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_splice")
    if cfg["max_clips"]:                              # local checks never overwrite the run
        cfg["out_dir"] = cfg["out_dir"].with_name(cfg["out_dir"].name + "_smoke")
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    print(f"mode={mode} device={cfg['device']} out={cfg['out_dir']} "
          f"max_clips={cfg['max_clips'] or 'all'} dose={cfg['dose']} "
          f"clip_source={cfg['clip_source']} splice={cfg['splice']}")
    assert cfg["dose"] == 1.0, "ablation only (brief §0)"
    if mode == "groups":
        run_groups(cfg)
    if mode == "check":
        run_check(cfg, dict(contrast_setup(cfg), splice=cfg["splice"]))
    if mode == "ablate":
        run_ablate(cfg, dict(contrast_setup(cfg), splice=cfg["splice"]))
    if mode == "summarise":
        run_summarise(cfg)
    if mode == "hand":
        run_hand(cfg)
    if mode == "all":
        run_groups(cfg)
        s = dict(contrast_setup(cfg), splice=cfg["splice"])
        run_check(cfg, s)
        run_ablate(cfg, s)
        run_summarise(cfg)
        run_hand(cfg)


if __name__ == "__main__":
    main()
