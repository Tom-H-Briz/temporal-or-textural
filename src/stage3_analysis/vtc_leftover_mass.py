"""Where does the attribution mass go after full ablation? (10/11/26) vtc_dose_sweep's two clips, L9,
spliced: grad x z of the pair margin (home - partner) per latent, summed over tokens, before and after
zeroing ALL that side's contrast passers. The features left carrying the margin are then read under
shuffle (VM C1, UMT C) and Rev on the same clip: activation and attribution (unablated). No controls.
Usage: uv run python src/stage3_analysis/vtc_leftover_mass.py   (CPU)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from vtc_dose_sweep import CFG as DS, ranked
CFG = dict(DS, shuffle={"videomae": "C1", "umt": "C"}, top_n=10,
           out_dir=ROOT / "outputs/analysis/vtc_leftover_mass_L9")


def splice_attr(st, pv, z, t, p):
    """Spliced state decode(z) + dim_mean at layer L; grad x z of logit(t) - logit(p), summed over
    tokens -> (n_lat,) attribution, and the margin itself."""
    from vision_pair_phase3 import forward_T
    z = z.clone().requires_grad_(True)
    T = forward_T(st, pv, (st["sae"].decode(z) + st["dm"]).unsqueeze(0), t, p)[0]
    return (torch.autograd.grad(T, z)[0] * z).sum(0).detach().cpu().numpy(), float(T)


def loader(model, st, clips, cond, cfg):
    """(pv, row) per clip under R / shuffle / Rev, each model's own sampler (Rev = R flipped in time)."""
    base = "R" if cond in ("R", "Rev") else cond
    if model == "videomae":
        from vm_object_contrast import clip_loader
        it = clip_loader(clips, cfg, base, st["processor"])
    else:
        from umt_mlm_feature_contrast import clip_loader
        it = clip_loader(clips, cfg, base)
    for pv, i in it:
        yield (pv.flip(1) if cond == "Rev" else pv).to(cfg["device"]), int(i)


def run_model(model, cfg):
    """{clip: {cond: (act, attr, margin)}} for R / shuffle / Rev (unablated) + 'R_full' (passers zeroed)."""
    import vision_pair_phase3
    st = vision_pair_phase3.setup(dict(cfg, model=model))
    clips = pd.DataFrame({"clip_id": list(cfg["clips"])})
    out = {c: {} for c in clips.clip_id}
    for cond in ("R", cfg["shuffle"][model], "Rev"):
        for pv, i in loader(model, st, clips, cond, cfg):
            c = clips.clip_id.iloc[i]
            pair, home = cfg["clips"][c]
            alt = [int(x) for x in pair.split("_") if int(x) != home][0]
            with torch.no_grad():
                st["model"](pixel_values=pv)          # hook fills st["x"]
                z0 = st["sae"].encode(st["x"][0].float() - st["dm"])[1]
            act = z0.sum(0).cpu().numpy()
            out[c][cond] = (act, *splice_attr(st, pv, z0, home, alt))
            if cond == "R":                           # all passers zeroed, same clip
                z = z0.clone(); z[:, ranked(model, pair, home, cfg).latent.astype(int).tolist()] = 0
                out[c]["R_full"] = (z.sum(0).cpu().numpy(), *splice_attr(st, pv, z, home, alt))
    return out


def tables(model, out, cfg):
    """Mass summary per clip (margin, passers' vs others' attribution, before / after) and the
    top_n features carrying the margin after full ablation, with their shuffle / Rev behaviour."""
    shuf, mass, top = cfg["shuffle"][model], [], []
    for c, d in out.items():
        pair, home = cfg["clips"][c]
        P = ranked(model, pair, home, cfg).latent.astype(int).to_numpy()
        a, af = d["R"][1], d["R_full"][1]
        mass.append(dict(model=model, clip_id=c, margin_R=d["R"][2], margin_full=d["R_full"][2],
                         attr_total_R=a.sum(), attr_passers_R=a[P].sum(), attr_total_full=af.sum(),
                         top10_share_full=np.sort(af)[::-1][:10].sum() / af[af > 0].sum()))
        for f in np.argsort(af)[::-1][:cfg["top_n"]]:
            top.append(dict(model=model, clip_id=c, latent=int(f), attr_R=a[f], attr_full=af[f],
                            act_R=d["R"][0][f], act_shuf=d[shuf][0][f], act_Rev=d["Rev"][0][f],
                            attr_shuf=d[shuf][1][f], attr_Rev=d["Rev"][1][f]))
    return mass, top


def main():
    cfg, mass, top = CFG, [], []
    for model in cfg["models"]:
        m, t = tables(model, run_model(model, cfg), cfg)
        mass += m; top += t
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    mass, top = pd.DataFrame(mass), pd.DataFrame(top)
    mass.to_csv(cfg["out_dir"] / "mass.csv", index=False); top.to_csv(cfg["out_dir"] / "top_left.csv", index=False)
    print(mass.round(3).to_string(index=False)); print(top.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
