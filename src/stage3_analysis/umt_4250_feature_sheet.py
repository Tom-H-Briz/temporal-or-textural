"""Fill UMT_4250_feature_analysis.csv (18/11/26): one row per L5 latent named in the 16/11 #4250 findings.
Categories from the real-clip PE probe (umt_pe_probe_real.py) + blank probe; '% of 4250' = linear attribution
(activation x gradient, raw run attr_R), NOT an ablation; logit column = MLM 'down' dCE (single-latent wordmap, spliced).
Usage: uv run python src/stage3_analysis/umt_4250_feature_sheet.py"""
import json, numpy as np, pandas as pd
from pathlib import Path
A = Path("outputs/analysis")
S = json.load(open("src/stage3_analysis/umt_clock_downstream_sets.json"))["5"]
CFG = {
    "groups": {"clock": S["clock"], "pe_set": S["pe"], "builders4": S["builders4"],
               "other_16_11": [5216, 515, 4535, 892, 5137, 3269, 4113, 768, 5006, 5281]},
    "real": A / "umt_pe_probe/real_43.npz", "blank": A / "umt_pe_probe/profiles.npz",
    "attr_run": A / "umt_clock_downstream_L5_raw", "target": (9, 4250),
    "mlm": A / "umt_mlm_feature_wordmap", "mlm_word": "down", "cls": 43,
    "synth_lift_max": 2.0,     # PE driven: blank >= real AND best synthetic < 2x blank (user, 18/11)
    "content_blank_max": 0.01, # content only: blank < 1% of real; otherwise mixed
    "out": A / "umt_clock_downstream_L5_2x2_pe_raw/analysis/UMT_4250_feature_analysis.csv",
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
    prof, tot, act, n = np.load(cfg["attr_run"] / "prof_L9_R.npy", mmap_mode="r"), 0.0, 0.0, 0
    for k in range(tg.shape[1]):
        idx = np.where((tg[:, k] == cfg["target"]).all(1))[0]
        if len(idx):
            tot = tot + at[idx, k, :, :-1].astype(np.float32).sum((0, 1))
            act, n = act + prof[idx, 0, :, cfg["target"][1]].astype(np.float32).sum(), n + len(idx)
    return 100 * tot / act                       # (sum / n) / (act / n)


def mlm_dce(cfg, f):
    """Mean dCE on the word 'down' (class cls clips) when latent f is ablated; NaN if no wordmap run."""
    p = cfg["mlm"] / f"L5_f{f}_splice/per_word.parquet"
    if not p.exists():
        return np.nan
    w = pd.read_parquet(p).query(f"word == '{cfg['mlm_word']}' and class_id == {cfg['cls']} and edit == 'single_{f}'")
    return w.dce.mean()


def main(cfg):
    grp = {}
    for g, fs in cfg["groups"].items():
        for f in fs:
            grp[f] = grp.get(f, []) + [g]
    F, at, cols = list(grp), attribution(cfg), ["PE driven", "Mixed (PE + content)", "Content only"]
    lv, rows = levels(cfg, F), []
    for i, f in enumerate(F):
        r = lv.iloc[i]
        cat = category(r, cfg)
        rows.append({"Feature Idx": f, **{c: "x" if c == cat else "" for c in cols},
                     "Blank act (PE alone)": round(float(r.blank)), "Real act (8 down clips)": round(float(r.real)),
                     "Best synthetic act": round(float(r.best_synth)), "Best synthetic clip": r.best_synth_clip,
                     "% of 4250 (linear attribution)": round(float(at[f]), 2),
                     "MLM 'down' dCE (single ablation)": round(mlm_dce(cfg, f), 4), "group": "+".join(grp[f])})
    out = pd.DataFrame(rows)
    out.to_csv(cfg["out"], index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main(CFG)
