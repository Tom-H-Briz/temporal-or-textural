"""VTC dose sweep (10/11/26): how many home-side pair features must be zeroed (spliced, L9, R)
before the class logit / pair margin moves? One clip per unit, VM and UMT side by side, no controls.
Features: that side's contrast passers, ranked by the mean of their AUROC rank and attribution rank;
a dose larger than the pool is capped at all passers (arm named by its real count).
Usage: uv run python src/stage3_analysis/vtc_dose_sweep.py   (CPU is fine: 2 clips x 5 arms per model)
"""
import os, sys
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
CFG = dict(layer=9, splice=True, dose=1.0, edit_batch=8, num_workers=0, doses=(3, 5, 10, 20, 25, 30, 999),   # 999 = all passers
           models=("videomae", "umt"),
           clips={"101394": ("168_165", 165), "180656": ("45_43", 43)},   # camera down, moving down
           video_dir=os.environ.get("VIDEO_DIR", str(ROOT / "data/ssv2/20bn-something-something-v2")),
           device="cuda" if torch.cuda.is_available() else "cpu",
           out_dir=ROOT / "outputs/analysis/vtc_dose_sweep_L9")


def ranked(model, pair, home, cfg):
    """One side's contrast passers, best first by mean(AUROC rank, attribution rank); ties -> AUROC."""
    c = pd.read_csv(ROOT / f"outputs/analysis/{model}_pair_contrast_L{cfg['layer']}/candidates_{pair}.csv")
    c = c[(c.home == home) & c.passed].copy()
    c["rank"] = (c.auroc.rank(ascending=False) + c.attr_mean.rank(ascending=False)) / 2
    return c.sort_values(["rank", "auroc"], ascending=[True, False])


def arms(model, pair, home, cfg):
    """[(name, feats)]: batch_base + one arm per dose (capped at the pool; duplicate caps dropped)."""
    c = ranked(model, pair, home, cfg).latent.astype(int).tolist()
    out = [("batch_base", [])]
    for k in sorted({min(d, len(c)) for d in cfg["doses"]}):
        out.append((f"dose_{k}", c[:k]))
    return out, len(c)


def run_model(model, cfg):
    """Both clips through one model: every dose arm's true logit, margin, pair margin (umt_object_vtc.clip_out)."""
    import vision_pair_phase3
    from umt_object_vtc import clip_out
    st = vision_pair_phase3.setup(dict(cfg, model=model))
    clips = pd.DataFrame({"clip_id": list(cfg["clips"])})
    if model == "videomae":                           # each model's own sampler + processor, as the VTC runs
        from vm_object_contrast import clip_loader
        loader = clip_loader(clips, cfg, "R", st["processor"])
    else:
        from umt_mlm_feature_contrast import clip_loader
        loader = clip_loader(clips, cfg, "R")
    rows = []
    for pv, i in loader:
        c = clips.clip_id.iloc[int(i)]
        pair, home = cfg["clips"][c]
        a, n_pool = arms(model, pair, home, cfg)
        alt = [int(x) for x in pair.split("_") if int(x) != home][0]
        rows += [dict(model=model, clip_id=c, unit=f"{pair}_{home}", n_pool=n_pool, **d)
                 for d in clip_out(st, pv.to(cfg["device"]), a, home, alt, dict(cfg, model=model))]
    return rows


def decoder_cos(cfg, n_rand=2000, seed=0):
    """Raw-space directions: cosine between the passers' decoder atoms (dose order), per model x unit,
    vs random atom pairs from the same dictionary. -> cos_<model>_<unit>.csv + summary rows."""
    from ToT_utils import MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    rows, g = [], torch.Generator().manual_seed(seed)
    for model in cfg["models"]:
        m = MODEL_REGISTRY[model]
        res = resolve_sae_checkpoint(model, cfg["layer"], dataset_name="ssv2", sae_k=64, job_label="7ep")
        D = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], "cpu", sae_k=64).get_dictionary().float()
        D = torch.nn.functional.normalize(D, dim=1)
        ri, rj = torch.randint(0, len(D), (2, n_rand), generator=g)
        rand = (D[ri] * D[rj]).sum(1).abs()
        for pair, home in cfg["clips"].values():
            f = ranked(model, pair, home, cfg).latent.astype(int).tolist()
            C = D[f] @ D[f].T
            pd.DataFrame(C.numpy(), index=f, columns=f).to_csv(cfg["out_dir"] / f"cos_{model}_{pair}_{home}.csv")
            rows.append(cos_summary(model, f"{pair}_{home}", C, rand))
    return pd.DataFrame(rows)


def cos_summary(model, unit, C, rand):
    """|cos| between passers (off-diagonal): median, max, share above the random 99th pct;
    and each feature in 11-all vs its nearest top-10 atom (is the 'later' pool a copy of the first?)."""
    n = len(C)
    off = C[~torch.eye(n, dtype=bool)].abs()
    p99 = float(rand.quantile(0.99))
    late = C[10:, :10].abs().max(1).values if n > 10 else torch.tensor([float("nan")])
    return dict(model=model, unit=unit, n_pool=n, rand_median=float(rand.median()), rand_p99=p99,
                median=float(off.median()), max=float(off.max()), pct_above_rand_p99=float((off > p99).float().mean() * 100),
                late_to_top10_median=float(late.median()), late_to_top10_max=float(late.max()))


def main():
    torch.backends.cuda.matmul.allow_tf32 = False     # fp32 paths, as the references
    cfg = CFG
    if os.environ.get("MODE") == "cos":               # decoder geometry only, no forward passes
        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        s = decoder_cos(cfg); s.to_csv(cfg["out_dir"] / "cos_summary.csv", index=False)
        return print(s.round(3).to_string(index=False))
    d = pd.DataFrame([r for m in cfg["models"] for r in run_model(m, cfg)])
    base = d[d.arm == "batch_base"].set_index(["model", "clip_id"])
    for m in ["logit_true", "margin", "pair_margin"]:
        d[f"d_{m}"] = d[m].values - base[m].reindex(pd.MultiIndex.from_frame(d[["model", "clip_id"]])).values
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    d.to_csv(cfg["out_dir"] / "dose_sweep.csv", index=False)
    cols = ["model", "clip_id", "unit", "n_pool", "arm", "logit_true", "d_logit_true", "pair_margin", "d_pair_margin", "correct"]
    print(d[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
