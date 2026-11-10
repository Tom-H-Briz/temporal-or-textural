"""Is VM's pair evidence one raw direction or many? (10/11/26) vtc_dose_sweep's two clips, L9, R, raw
residual (no SAE in the edit). Readout direction = grad of the pair margin (home - partner) wrt the L9
patch tokens, summed over tokens, unit norm. Removed from every token around dim_mean (the variation
along it goes, the mean stays); the gradient is then recomputed on the edited stream and the next
direction removed (orthogonal to the earlier ones): margin after k = 1, 2, 3, 5, 10, 20 directions.
Also: |cos| of each passer's decoder atom with the first direction vs random atoms. No controls.
Usage: uv run python src/stage3_analysis/vtc_readout_direction.py   (CPU)
"""
import sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from vtc_dose_sweep import CFG as DS, ranked
from vtc_leftover_mass import loader
CFG = dict(DS, ks=(1, 2, 3, 5, 10, 20), n_rand=2000, out_dir=ROOT / "outputs/analysis/vtc_readout_direction_L9")


def margin_grad(st, pv, x, t, p):
    """Pair margin with x (N, D) spliced at layer L, and its gradient wrt x summed over tokens (D,)."""
    from vision_pair_phase3 import forward_T
    x = x.clone().requires_grad_(True)
    T = forward_T(st, pv, x.unsqueeze(0), t, p)[0]
    return float(T), torch.autograd.grad(T, x)[0].sum(0)


def remove_dirs(st, pv, x, t, p, cfg):
    """Strip readout directions one at a time (recomputed after each) -> margins at k, and Q (K, D)."""
    Q, out, xk = torch.zeros(0, x.shape[1]), {}, x
    T0, g = margin_grad(st, pv, x, t, p)
    for k in range(1, max(cfg["ks"]) + 1):
        g = g - Q.T @ (Q @ g)                         # orthogonal to directions already removed
        Q = torch.cat([Q, (g / g.norm())[None]])
        xk = x - ((x - st["dm"]) @ Q.T) @ Q           # variation along Q gone, dim_mean kept
        T, g = margin_grad(st, pv, xk, t, p)
        if k in cfg["ks"]:
            out[k] = T
    return T0, out, Q


def atom_cos(st, q1, model, pair, home, cfg):
    """|cos| of passers' decoder atoms (dose order) with the first readout direction, vs random atoms."""
    D = torch.nn.functional.normalize(st["D"].float().cpu(), dim=1)
    P = ranked(model, pair, home, cfg).latent.astype(int).tolist()
    c = (D @ q1).abs()
    r = c[torch.randint(0, len(D), (cfg["n_rand"],), generator=torch.Generator().manual_seed(0))]
    top = torch.topk(c, 10).indices.tolist()          # atoms best aligned with the readout, any feature
    return dict(pass_median=float(c[P].median()), pass_max=float(c[P].max()), pass_sumsq=float((c[P] ** 2).sum()),
                rand_median=float(r.median()), rand_p99=float(r.quantile(0.99)),
                top10_atoms=top, top10_cos=[round(float(c[i]), 3) for i in top],
                top10_in_passers=sum(i in P for i in top))


def main():
    import vision_pair_phase3
    cfg, rows, cos = CFG, [], []
    clips = pd.DataFrame({"clip_id": list(cfg["clips"])})
    for model in cfg["models"]:
        st = vision_pair_phase3.setup(dict(cfg, model=model))
        for pv, i in loader(model, st, clips, "R", cfg):
            c = clips.clip_id.iloc[i]
            pair, home = cfg["clips"][c]
            alt = [int(x) for x in pair.split("_") if int(x) != home][0]
            with torch.no_grad():
                st["model"](pixel_values=pv)          # hook fills st["x"]
                x = st["x"][0].float()
            T0, m, Q = remove_dirs(st, pv, x, home, alt, cfg)
            rows += [dict(model=model, clip_id=c, k=0, margin=T0)] + [dict(model=model, clip_id=c, k=k, margin=v) for k, v in m.items()]
            cos.append(dict(model=model, clip_id=c, **atom_cos(st, Q[0], model, pair, home, cfg)))
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    r, cs = pd.DataFrame(rows), pd.DataFrame(cos)
    r.to_csv(cfg["out_dir"] / "margins.csv", index=False); cs.to_csv(cfg["out_dir"] / "atom_cos.csv", index=False)
    print(r.pivot_table(index=["model", "clip_id"], columns="k", values="margin").round(2).to_string()); print(cs.to_string(index=False))


if __name__ == "__main__":
    main()
