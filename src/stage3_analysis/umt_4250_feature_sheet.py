"""Feature sheet for one target (18/11/26; TARGET = a PRESETS key: 4250, ball_713). Rows = top K L5 latents by
|linear attribution| to the target, sign from the data (user rule, 18/11: portable to VideoMAE; the #4250 run used
20 pos + 20 neg, a superset). Buckets from blank / real / best synthetic; ablation columns from the group-ablation
run(s); MLM column = dCE on the preset's word (single-latent wordmap, spliced).
WRITE_SETS=1 also writes the bucket x sign groups to the sets json (key top20_<TARGET>) for the ARM_GROUPS run.
Usage: [TARGET=ball_713] [WRITE_SETS=1] uv run python src/stage3_analysis/umt_4250_feature_sheet.py"""
import json, os, numpy as np, pandas as pd
from pathlib import Path
A = Path("outputs/analysis")
PRESETS = {   # per target (TARGET env); attr_roles = clip roles averaged in the attribution (None = all clips)
    "4250": dict(target=(9, 4250), real=A / "umt_pe_probe/real_43.npz", real_label="8 down clips",
                 attr_run=A / "umt_clock_downstream_L5_raw", attr_roles=None, mlm_word="down", mlm_cls=43,
                 abl=[A / "umt_top20_4250_L5_ud_splice/analysis/group_ablation.csv",   # first found wins per feature
                      A / "umt_group_ablation_L5_ud_splice/analysis/group_ablation.csv"],
                 out=A / "umt_clock_downstream_L5_2x2_pe_raw/analysis/UMT_4250_feature_analysis.csv"),
    "ball_713": dict(target=(9, 713), real=A / "umt_pe_probe/real_ball.npz", real_label="8 ball clips",
                     attr_run=A / "umt_noun_attr_ball_L5_raw", attr_roles=["pos", "test"], mlm_word="ball", mlm_cls=None, mlm_suffix="_ball",
                     abl=[A / "umt_top20_ball_713_L5_splice/analysis/group_ablation.csv"],
                     out=A / "umt_noun_attr_ball_L5_raw/analysis/UMT_ball_713_feature_analysis.csv"),
}
TARGET = os.environ.get("TARGET", "4250")
CFG = {
    **PRESETS[TARGET], "top_k": 20, "sets_key": f"top20_{TARGET}", "write_sets": os.environ.get("WRITE_SETS", "") == "1",
    "sets": Path("src/stage3_analysis/umt_clock_downstream_sets.json"), "blank": A / "umt_pe_probe/profiles.npz",
    "mlm": A / "umt_mlm_feature_wordmap",
    "synth_lift_max": 2.0,     # PE driven: blank >= real AND best synthetic < 2x blank (user, 18/11)
    "content_blank_max": 0.01, # content only: blank < 1% of real; otherwise mixed
}


def levels(cfg, F):
    """Per row: blank (PE alone), real (8 down clips) and best synthetic activation + its clip; frame-summed,
    ordinary tokens. Synthetic token maps via umt_4250_feature_checks.synth_tokens."""
    from umt_4250_feature_checks import CFG as KCFG, synth_tokens
    real = np.load(cfg["real"])["full"].astype(np.float32).mean(0)[:, F].sum(0)
    blank = np.load(cfg["blank"])["blank|full"].astype(np.float32)[:, F].sum(0)
    act, _, _, names, _ = synth_tokens(KCFG, F)
    lev = act.sum((1, 2))                                                                  # (clips, n)
    return pd.DataFrame(dict(blank=blank, real=real, best_synth=lev.max(0), best_synth_clip=names[lev.argmax(0)]))


def category(r, cfg):
    """PE driven / Mixed / Content only from raw blank, real and best synthetic activation."""
    if r.blank < cfg["content_blank_max"] * r.real:
        return "Content only"
    if r.blank >= r.real and r.best_synth < cfg["synth_lift_max"] * r.blank:
        return "PE driven"
    return "Mixed (PE + content)"


def attribution(cfg):
    """(F,) clip-summed attribution of the target to each L5 latent as % of the target's mean clip-summed activation."""
    at, tg = (np.load(cfg["attr_run"] / f"{n}_R.npy", mmap_mode="r") for n in ("attr", "attr_tg"))
    prof = np.load(cfg["attr_run"] / f"prof_L{cfg['target'][0]}_R.npy", mmap_mode="r")
    keep, tot, act, n = np.ones(len(tg), bool), 0.0, 0.0, 0
    if cfg["attr_roles"]:                        # nouns: object clips only (no-object clips are the comparison)
        roles = pd.read_csv(cfg["attr_run"] / "clips.csv", dtype={"clip_id": str}).role
        keep = roles.isin(cfg["attr_roles"]).to_numpy()
    for k in range(tg.shape[1]):
        idx = np.where((tg[:, k] == cfg["target"]).all(1) & keep)[0]
        if len(idx):
            tot = tot + at[idx, k, :, :-1].astype(np.float32).sum((0, 1))
            act, n = act + prof[idx, 0, :, cfg["target"][1]].astype(np.float32).sum(), n + len(idx)
    return 100 * tot / act                       # (sum / n) / (act / n)


def ablation(cfg):
    """{feature: row} from the group-ablation runs' single arms (R, spliced); earlier runs in cfg['abl'] win."""
    out = {}
    for p in [p for p in cfg["abl"] if p.exists()][::-1]:          # later list entries first, overwritten by earlier
        a = pd.read_csv(p).query("cond == 'R' and arm.str.startswith('f')")
        out.update({int(r.arm[1:]): r for _, r in a.iterrows()})
    return out


def mlm_dce(cfg, f):
    """Mean dCE on mlm_word (on mlm_cls clips, or every clip if None) when latent f is ablated; NaN if no wordmap run."""
    p = cfg["mlm"] / f"L5_f{f}{cfg.get('mlm_suffix', '')}_splice/per_word.parquet"
    if not p.exists():
        return np.nan
    w = pd.read_parquet(p).query(f"word == '{cfg['mlm_word']}' and edit == 'single_{f}'")
    return (w if cfg["mlm_cls"] is None else w[w.class_id == cfg["mlm_cls"]]).dce.mean()


def abl_cols(r):
    """Readable ablation columns from one group_ablation row: every pct_L{c}_{t} and dmargin_{cls} it has."""
    out = {}
    for k, v in r.items():
        if k.startswith("pct_L"):
            out[f"% of {k.split('_')[2]} removed (ablation, R)"] = round(float(v), 2)
        elif k.startswith("dmargin_"):
            out[f"d margin on {k.split('_')[1]} clips (ablation, R)"] = round(float(v), 3)
    return out


def write_sets(cfg, out):
    """{bucket_sign: feats} for every bucket x {pos, neg} plus each bucket net -> sets json [5][sets_key]."""
    key = {"PE driven": "pe", "Mixed (PE + content)": "mixed", "Content only": "content"}
    s, g = json.load(open(cfg["sets"])), {}
    for b, k in key.items():
        m = out[out[b] == "x"]
        for sign in ("pos", "neg"):
            g[f"{k}_{sign}"] = m.loc[m.group == sign, "Feature Idx"].astype(int).tolist()
        g[f"{k}_net"] = m["Feature Idx"].astype(int).tolist()
    s["5"][cfg["sets_key"]] = {k: v for k, v in g.items() if v}         # empty groups dropped
    json.dump(s, open(cfg["sets"], "w"))
    print({k: len(v) for k, v in g.items()})


def main(cfg):
    at = attribution(cfg)                        # top K by |attribution|, sign from the data (user rule, 18/11)
    grp = {int(f): "pos" if at[f] > 0 else "neg" for f in np.argsort(-np.abs(at))[:cfg["top_k"]]}
    F, cols = list(grp), ["PE driven", "Mixed (PE + content)", "Content only"]
    lv, rows, ab, t = levels(cfg, F), [], ablation(cfg), cfg["target"][1]
    for i, f in enumerate(F):
        r = lv.iloc[i]
        cat = category(r, cfg)
        rows.append({"Feature Idx": f, **{c: "x" if c == cat else "" for c in cols},
                     "Blank act (PE alone)": round(float(r.blank)), f"Real act ({cfg['real_label']})": round(float(r.real)),
                     "Best synthetic act": round(float(r.best_synth)), "Best synthetic clip": r.best_synth_clip,
                     **(abl_cols(ab[f]) if f in ab else {}),
                     f"% of {t} (linear attribution)": round(float(at[f]), 2),
                     f"MLM '{cfg['mlm_word']}' dCE (single ablation)": round(mlm_dce(cfg, f), 4), "group": grp[f]})
    out = pd.DataFrame(rows)                     # features missing from the ablation run get NaN in those columns
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg["out"], index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    if cfg["write_sets"]:
        write_sets(cfg, out)


if __name__ == "__main__":
    main(CFG)
