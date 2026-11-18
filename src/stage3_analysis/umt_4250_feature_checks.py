"""Separation check (18/11/26) for the #4250 sheet tests, existing data only: do timing x level scores split the known
pure-PE / mixed / content L5 latents? Level: Shapley over PE {on, off} x content {real, blank}. Timing: mirror, shuffle
shape. Content agnostic: shuffle level, real class spread, synthetic spread, object vs background tokens; top synthetic clip.
Usage: uv run python src/stage3_analysis/umt_4250_feature_checks.py"""
import json, numpy as np, pandas as pd
from pathlib import Path
A = Path("outputs/analysis")
CFG = {
    "known": {"pure_PE": [3453, 5872, 3718, 5387, 5122, 3829, 1004, 966, 5430, 4183, 892, 5137],
              "mixed": [5422, 5286, 2377], "content": [186, 4638, 5216]},
    "sheet": A / "umt_clock_downstream_L5_2x2_pe_raw/analysis/UMT_4250_feature_analysis.csv",
    "real": A / "umt_pe_probe/real_43.npz", "blank": A / "umt_pe_probe/profiles.npz",
    "shuffle": A / "umt_layer_positional_ablation_L5_splice/profiles_{c}.npz",
    "synth": [A / "umt_synthetic_motion_readout", A / "umt_synthetic_pair_readout", A / "umt_synthetic_pair_color_readout"],
    "massive_norm": 200.0, "patch": 16, "obj_r": 1, "bg_r": 3,   # object = 3x3 patches round centre; bg > 3 patches away
    "out": A / "umt_4250_feature_checks",
}


def corr(a, b):
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-9)


def level_2x2(cfg, F):
    """Shapley split of (real full - blank no-PE) into PE, content and their interaction; plus mirror and frame CV."""
    d = {k: v.astype(np.float32).mean(0)[:, F] for k, v in np.load(cfg["real"]).items()}   # (12, n) clip means
    b = np.load(cfg["blank"])
    RF, RN = d["full"].sum(0), d["none"].sum(0)
    BF, BN = (b[f"blank|{c}"].astype(np.float32)[:, F].sum(0) for c in ("full", "none"))
    tot = np.where(np.abs(RF - BN) > 1e-6, RF - BN, np.nan)
    inter = RF - RN - BF + BN
    return pd.DataFrame(dict(
        feature=F, real=RF, pe_share=((RF - RN) + (BF - BN)) / 2 / tot, content_share=((RF - BF) + (RN - BN)) / 2 / tot,
        interaction=inter / tot, blank_over_real=BF / RF, mirror=corr(d["full"][::-1].T, d["reversed"].T),
        frame_cv=d["full"].std(0) / (d["full"].mean(0) + 1e-9),
        frozen_over_full=b["blank|frozen"].astype(np.float32)[:, F].sum(0) / (BF + 1e-9)))


def shuffle(cfg, F):
    """1,400 real clips, R vs shuffled C: level C/R, median per-clip shape r (clips active in both), class-level CV."""
    R, C = (np.load(str(cfg["shuffle"]).format(c=c)) for c in ("R", "C"))
    pr, pc = R["prof"][:, :, F].astype(np.float32), C["prof"][:, :, F].astype(np.float32)   # (clips, 12, n)
    lr, lc = pr.sum(1), pc.sum(1)
    r = corr(pr.transpose(0, 2, 1), pc.transpose(0, 2, 1))                                 # (clips, n)
    r[(pr.std(1) == 0) | (pc.std(1) == 0)] = np.nan
    cls = pd.DataFrame(lr).groupby(R["class_id"]).mean()
    return pd.DataFrame(dict(shuffle_level=lc.sum(0) / (lr.sum(0) + 1e-9), shuffle_shape_r=np.nanmedian(r, 0),
                             class_cv=(cls.std() / (cls.mean() + 1e-9)).values, frac_clips_on=(lr > 0).mean(0)))


def synth_tokens(cfg, F):
    """All synthetic clips: per-token activation (clips, 12, 196, n) over ordinary tokens, object / background masks
    (clips, 12, 196) from the trajectories, clip names and kind (single / pair)."""
    act, obj, bg, names, kind = [], [], [], [], []
    rc = np.stack(np.divmod(np.arange(196), 14), 1)                                        # (196, 2) row, col
    for d in cfg["synth"]:
        idx, val = np.load(d / "idx_L5.npy"), np.load(d / "val_L5.npy").astype(np.float32)
        val = val * (np.load(d / "norm_L5.npy") < cfg["massive_norm"])[..., None]
        act.append(np.stack([(val * (idx == f)).sum(-1) for f in F], -1))
        for m in json.load(open(d / "metadata.json")):
            xy = np.array(m["umt_trajectory_xy"], dtype=float).reshape(12, -1, 2) // cfg["patch"]   # (12, objs, x/y)
            dist = np.abs(rc[None, :, None, :] - xy[:, None, :, ::-1]).max(-1).min(-1)          # (12, 196) Chebyshev
            obj.append(dist <= cfg["obj_r"]); bg.append(dist > cfg["bg_r"])
            names.append(d.name.replace("umt_synthetic_", "").replace("_readout", "") + ":" + m["name"])
            kind.append("pair" if "pair" in d.name else "single")
    return np.concatenate(act), np.stack(obj), np.stack(bg), np.array(names), np.array(kind)


def synthetic(cfg, F, lv):
    """Content dose (blank -> one object -> two objects -> real), spread, shape vs the blank (PE) profile, object vs
    background token mean, top clip as a ratio to blank. lv: level_2x2 table (blank level = blank_over_real * real)."""
    act, obj, bg, names, kind = synth_tokens(cfg, F)
    prof, lev = act.sum(2), act.sum((1, 2))                                                # (clips, 12, n), (clips, n)
    blank_p = np.load(cfg["blank"])["blank|full"].astype(np.float32)[:, F]                # (12, n)
    blank = (lv.blank_over_real * lv.real).values
    tok = lambda m: (act * m[..., None]).sum((0, 1, 2)) / m.sum()
    top = lev.argmax(0)
    return pd.DataFrame(dict(
        single_over_blank=lev[kind == "single"].mean(0) / (blank + 1e-9), pair_over_blank=lev[kind == "pair"].mean(0) / (blank + 1e-9),
        real_over_blank=1 / (lv.blank_over_real.values + 1e-9), synth_cv=lev.std(0) / (lev.mean(0) + 1e-9),
        synth_shape_vs_blank=np.nanmedian(corr(prof.transpose(0, 2, 1), blank_p.T[None]), 0),
        obj_over_bg=tok(obj) / (tok(bg) + 1e-9), top_clip=names[top], top_over_blank=lev.max(0) / (blank + 1e-9)))


def main(cfg):
    F = pd.read_csv(cfg["sheet"], encoding="utf-8-sig")["Feature Idx"].tolist()
    known = {f: k for k, fs in cfg["known"].items() for f in fs}
    lv = level_2x2(cfg, F)
    out = pd.concat([lv, shuffle(cfg, F), synthetic(cfg, F, lv)], axis=1)
    out.insert(1, "known", [known.get(f, "") for f in F])
    cfg["out"].mkdir(exist_ok=True)
    out.to_csv(cfg["out"] / "checks.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.sort_values("known").drop(columns="top_clip").round(2).to_string(index=False))
    print(out.sort_values("known")[["feature", "known", "top_clip"]].to_string(index=False))


if __name__ == "__main__":
    main(CFG)
