"""27/11/26: formal PE time-sensitivity spectrum — one committed replacement for the exploratory chain in
outputs/analysis/pe_groups_251126/ (sweep_real -> spectrum -> spectrum_shape -> spectrum_rank -> spectrum_alltok,
several of which overwrite their input). Per model x layer, on the C75 latents, ALL tokens (ordinary + <key>_massive):
size = worst k of |log2 frozen/real| clip-mean totals after dividing out the layer's own mass-per-firing-latent shift;
mag_lost = real frame-profile CV - min over k of the frozen CV; pct = within-layer percentile of the mean of the two
percentiles; seg = five 20% segments s1 (most time-sensitive) .. s5. ap = sum over classes of class-mean
|signed_vec_R| from the all-174 thrfix AP parquets (attribution patching; the "dfa" name on file). Checks vs
spectrum_alltok.parquet and arms_mass.csv picks run before anything uses the new file.
Usage: uv run python src/stage3_analysis/pe_spectrum.py"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
from umt_layer_specificity import CFG as SPEC
from frozen_pe_real_analyse import rows

A = ROOT / "outputs/analysis"
CFG = {
    "models": ("umt", "vm", "tf"), "layers": (5, 7, 9),
    "n_k": {"umt": 12, "vm": 8, "tf": 8},
    "ap": {"umt": str(SPEC["dfa"]),
           "vm": str(A / "dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_ssv2_all174_thrfix_l{}_job7ep_k64.parquet"),
           "tf": str(A / "dfa_mass_delta_tf/dfa_mass_delta_tf_ssv2_all174_thrfix_l{0}_job{0}_k64.parquet")},
    "sweep_dir": str(A / "{}_layer_pe_characterise"), "r_floor": 1e-3, "c_mass": 0.75,
    "c75": A / "pe_groups_251126/c75_latents.parquet",
    "ref": A / "pe_groups_251126/spectrum_alltok.parquet",
    "arms": A / "pe_groups_251126/arms_mass.csv",
    "out": A / "pe_spectrum/spectrum.parquet", "tol": 1e-6,
    "check": os.environ.get("CHECK", "1") == "1",   # CHECK=0: skip the checks vs the exploratory pe_groups_251126 files
}


def load_ap(path, L):
    """(174, F) per-class mean |signed_vec_R| from an all-174 thrfix AP parquet (filters_c.py)."""
    d = pd.read_parquet(str(path).format(L), columns=["class_id", "signed_vec_R"])
    return pd.DataFrame(np.abs(np.stack(d.signed_vec_R.to_numpy()))).groupby(d.class_id.values).mean().to_numpy()


def c75_sets():
    """C75 = latents in the top 75% of |AP| mass for >= 1 class, recomputed; stops on mismatch with the file."""
    ref = pd.read_parquet(CFG["c75"]) if CFG["check"] else None; out = {}
    for m in CFG["models"]:
        for L in CFG["layers"]:
            mabs = load_ap(CFG["ap"][m], L); top = np.zeros(mabs.shape[1], bool)
            for row in mabs:
                o = np.argsort(-row); cs = np.cumsum(row[o]) / row.sum()
                top[o[:int((cs < CFG["c_mass"]).sum() + 1)]] = True
            out[(m, L)] = np.where(top)[0]
            if ref is None:
                continue
            have = sorted(ref[(ref.model == m) & (ref.layer == L)].latent.tolist())
            if out[(m, L)].tolist() != have:
                sys.exit(f"STOP: C75 mismatch {m} L{L} (recomputed {top.sum()} vs file {len(have)})")
    return out


def per_lat(x):
    """Layer mass per firing latent-clip: total mass / firing count (spectrum_alltok.py)."""
    t = x.sum(1); return t.sum() / (t > 0).sum()


def frame_cv(x):
    """Frame CV of the clip-mean profile (spectrum_shape.py)."""
    p = x.mean(0); return p.std(0) / p.mean(0)


def load_cond(m, L, k):
    """(clip ids, (clips, frames, F) float32, layer shift) for R (k None) or frozen at k, all tokens."""
    d = Path(str(CFG["sweep_dir"]).format(m))
    pre, key = ("", "R") if k is None else (f"frozen{k or ''}_", f"R_frozen{k or ''}")
    z, z5 = np.load(d / f"{pre}L{L}.npz"), np.load(d / f"{pre}L5.npz")
    x = z[key].astype(np.float32)
    if f"{key}_massive" in z.files:
        x = x + z[f"{key}_massive"].astype(np.float32)
    return list(rows(z, z5, key)), x, per_lat(x)


def sweep_layer(m, L, lat, ap):
    """Size = worst k of |log2 frozen/real| after the layer shift; mag_lost = real CV - min frozen CV, per C75 latent."""
    ids, R, base = load_cond(m, L, None); Rs = R[:, :, lat]; del R
    runs = []
    for k in range(CFG["n_k"][m]):
        i, F, lay = load_cond(m, L, k); runs.append((i, F[:, :, lat], lay / base)); del F
    com = sorted(set(ids).intersection(*[set(r[0]) for r in runs]))
    pick = lambda arr, r: arr[[r.index(c) for c in com]]
    Rt = pick(Rs, ids).sum(1).mean(0)
    W, fcv = [], []
    with np.errstate(all="ignore"):
        for i, F, infl in runs:
            Ft = pick(F, i).sum(1).mean(0)
            W.append(np.abs(np.log2(np.clip(Ft / Rt, CFG["r_floor"], None)) - np.log2(infl)))
            fcv.append(frame_cv(F))
        c0 = frame_cv(Rs)
        df = pd.DataFrame(dict(model=m, L=L, latent=lat, ap=ap, size=np.nanmax(W, 0),
                               mag_lost=c0 - np.nanmin(fcv, 0), n_clips=len(com)))
    print(f"{m} L{L}: n {len(lat)} clips {len(com)} shift/k "
          f"{np.round([r[2] for r in runs], 3).tolist()}", flush=True)
    return df


def rank_seg(S):
    """pct = within-layer percentile of the mean size / mag-lost percentiles; seg = five 20% bands (arms_mass.py)."""
    out = []
    for _, g in S.groupby(["model", "L"], sort=False):
        g = g.copy()
        g["pct"] = ((g["size"].rank(pct=True) + g["mag_lost"].rank(pct=True)) / 2).rank(pct=True)
        seg = pd.Series("", index=g.index)
        for s in range(5):
            seg[(g.pct > 1 - (s + 1) / 5) & (g.pct <= 1 - s / 5)] = f"s{s + 1}"
        g["seg"] = seg
        out.append(g)
        print(f"   segments: {g.seg.value_counts().sort_index().to_dict()}", flush=True)
    return pd.concat(out, ignore_index=True)


def check_ap(ap):
    """Recomputed AP must equal the dfa column on file (spectrum_alltok.parquet) for all three models; else stop."""
    ref = pd.read_parquet(CFG["ref"]); bad = []
    for (m, L), g in ref.groupby(["model", "L"], sort=False):
        d = np.abs(ap[(m, L)][g.latent.values] - g.dfa.values); mx = float(np.max(d))
        print(f"{m} L{L} max|ap - dfa on file| {mx:.3e}", flush=True)
        if mx > 0:
            bad.append((m, L, mx))
    if bad:
        for m, L, mx in bad:
            print(f"STOP: ap != dfa on file for {m} L{L} (max {mx:.3e}) — came from features_L{{L}}.csv; don't pick one")
        sys.exit(1)


def check_spectrum(S):
    """vs spectrum_alltok.parquet: same latents; size/mag_lost/pct = Wn_all/cv_lost_all/pct_all (< tol); same segs."""
    ref = pd.read_parquet(CFG["ref"]); ok = True
    for (m, L), g in S.groupby(["model", "L"], sort=False):
        r = ref[(ref.model == m) & (ref.L == L)].set_index("latent").loc[g.latent.values]
        seg = pd.Series("", index=r.index)
        for s in range(5):
            seg[(r.pct_all > 1 - (s + 1) / 5) & (r.pct_all <= 1 - s / 5)] = f"s{s + 1}"
        d = {a: float(np.nanmax(np.abs(g[a].values - r[b].values)))
             for a, b in (("size", "Wn_all"), ("mag_lost", "cv_lost_all"), ("pct", "pct_all"))}
        print(f"{m} L{L} max|diff| " + " ".join(f"{k} {v:.2e}" for k, v in d.items())
              + f" | latents {len(g)} vs ref {len(r)} | seg mismatches {int((g.seg.values != seg.values).sum())}",
              flush=True)
        ok &= set(g.latent) == set(r.index) and max(d.values()) < CFG["tol"] and (g.seg.values == seg.values).all()
    return bool(ok)


def check_arms(S):
    """Every latent picked in arms_mass.csv (the arms the ablation ran on) must sit in the same segment here."""
    arms = pd.read_csv(CFG["arms"]); ix = S.set_index(["model", "L", "latent"]).seg; bad = n = 0
    for _, r in arms.iterrows():
        for f in map(int, str(r.picks).split()):
            n += 1; bad += ix.loc[(r.model, r.L, f)] != r.seg
    print(f"arms_mass.csv: {n} picks, {bad} segment mismatches", flush=True)
    return bad == 0


def main():
    lat = c75_sets()
    ap = {(m, L): load_ap(CFG["ap"][m], L).sum(0) for m in CFG["models"] for L in CFG["layers"]}
    if CFG["check"]:
        check_ap(ap)
    S = rank_seg(pd.concat([sweep_layer(m, L, lat[(m, L)], ap[(m, L)][lat[(m, L)]])
                            for m in CFG["models"] for L in CFG["layers"]], ignore_index=True))
    S = S[["model", "L", "latent", "ap", "size", "mag_lost", "pct", "seg", "n_clips"]]
    if CFG["check"] and not (check_spectrum(S) and check_arms(S)):
        sys.exit(f"CHECKS FAILED — not writing {CFG['out']}")
    CFG["out"].parent.mkdir(parents=True, exist_ok=True)
    S.to_parquet(CFG["out"])
    print(f"wrote {CFG['out']} ({len(S)} rows)")


if __name__ == "__main__":
    main()
