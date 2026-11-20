"""VideoMAE down decision in pure PE / mixed / content terms (20/11/26), the parts of the #4250 analysis that need no
single target feature: (1) down-class DFA by group per layer, R vs C1 (umt_4250_breakdown.dfa_table on VideoMAE's
trained-threshold all-174 DFA); (2) dictionary-wide DFA mass by group (active latents, classes weighted equally);
(3) activation vs use on down clips. Groups from vm_layer_pe_characterise (same rule as UMT).
Out: outputs/analysis/vm_down_breakdown/. Usage: uv run python src/stage3_analysis/vm_down_breakdown.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
import umt_4250_breakdown as ub

A = ROOT / "outputs/analysis"
CFG = dict(ub.CFG, model="VideoMAE", pc=A / "vm_layer_pe_characterise", conds=("R", "C1"), out=A / "vm_down_breakdown",
           dfa=A / "dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1_ssv2_all174_thrfix_l{}_job7ep_k64.parquet")


def dict_dfa_mass(cfg):
    """Per layer and group: share of class-weighted |DFA| (R) over active latents, as UMT's pct_dfa_mass_active."""
    rows = []
    for L in cfg["layers"]:
        d = pd.read_parquet(str(cfg["dfa"]).format(L), columns=["class_id", "signed_vec_R"])
        X = pd.DataFrame(np.abs(np.stack(d.signed_vec_R.to_numpy())).astype(np.float32)).groupby(d.class_id.values).mean()
        mass, (b, f) = X.mean(0).to_numpy(), ub.buckets(L, cfg)
        act = f.fire_rate.to_numpy() >= cfg["active"]
        for k in cfg["buckets"]:
            rows.append(dict(layer=L, bucket=k, pct_dfa_mass_active=100 * mass[act & (b == k)].sum() / mass[act & np.isin(b, cfg["buckets"])].sum()))
    return pd.DataFrame(rows)


def activation_vs_use(cfg):
    """Down clips, per layer and group: share of activation vs |DFA|. Activation: L7 / L9 = 311 down clips (VM L5 raw
    run, baseline arm); L5 = the 4 down clips of the characterisation run (no larger L5 down set stored)."""
    D = A / "vm_clock_downstream_L5_raw"
    idx = np.where(pd.read_csv(D / "clips.csv").class_id.to_numpy() == cfg["cls"])[0]
    pc = pd.read_csv(cfg["pc"] / "clips.csv").class_id.to_numpy() == cfg["cls"]
    act = {5: np.load(cfg["pc"] / "L5.npz")["R"][pc].astype(np.float64).sum(1).mean(0)}
    for L in (7, 9):
        act[L] = np.load(D / f"prof_L{L}_R.npy", mmap_mode="r")[idx, 0].astype(np.float64).sum(1).mean(0)
    rows = []
    for L in cfg["layers"]:
        b, v = ub.buckets(L, cfg)[0], ub.down_dfa(L, "R", cfg)[0]
        for k in cfg["buckets"]:
            m = b == k
            rows.append(dict(layer=L, bucket=k, n_act_clips=int(pc.sum()) if L == 5 else len(idx),
                             pct_activation=100 * act[L][m].sum() / act[L].sum(), pct_abs_dfa=100 * np.abs(v[m]).sum() / np.abs(v).sum()))
    return pd.DataFrame(rows)


def main(cfg):
    cfg["out"].mkdir(parents=True, exist_ok=True)
    t, dm, av = ub.dfa_table(cfg), dict_dfa_mass(cfg), activation_vs_use(cfg)
    for n, x in (("down_dfa_by_layer", t), ("dict_dfa_mass", dm), ("activation_vs_use", av)):
        x.round(4).to_csv(cfg["out"] / f"{n}.csv", index=False)
    ub.plot_dfa(t, cfg); ub.plot_act_vs_use(av, cfg)


if __name__ == "__main__":
    main(CFG)
