"""
L3 PE lineage (16/11/26): are the L5 content-boosted clock ramps (#5422, #2377, #5286) children of PE timing at L3?
No UMT L3 SAE, so the basis is the PE itself. Real down / up clips (43 / 45). (1) PE probe on real clips: L5
profiles (ordinary tokens) under full / frozen / reversed PE. (2) Attribution at L3: gradient of each L5 latent's
clip sum w.r.t. the L3 residual, attribution x.g split into the PE time subspace (per-frame PE means, rank 11),
the PE space subspace (frame-0 patch codes) and the rest. Comparisons: pure-PE clock, builders.
Usage: MAX_CLIPS=12 uv run python -u src/stage3_analysis/umt_l3_pe_lineage.py
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from umt_mlm_group_ablation import CFG as GA_CFG
from umt_pe_probe import pe_variant

CFG = dict(GA_CFG, layer=5, src_layer=3, classes=[43, 45], max_clips=int(os.environ.get("MAX_CLIPS", 12)),
           num_workers=0, massive_norm=200.0, pe_conds=["full", "frozen", "reversed"],
           feats={5422: "content clock", 2377: "content clock", 5286: "content clock",
                  3453: "PE clock", 5872: "PE clock", 4638: "builder", 186: "builder"},
           dfa=ROOT / "outputs/analysis/dfa_mass_delta_umt/dfa_mass_delta_umt_ssv2_all174_thrfix_l9_job7ep_k64.parquet",
           out_dir=ROOT / "outputs/analysis/umt_l3_pe_lineage")


def pe_bases(pe):
    """Orthonormal bases (k, 768): time = per-frame PE means centred over frames (rank 11); space = frame-0
    patch codes centred over patches, top components holding 99% of their energy."""
    p = pe[0].reshape(12, 196, -1).float()
    t = p.mean(1) - p.mean((0, 1))
    _, s, vt = torch.linalg.svd(t, full_matrices=False)
    time = vt[: int((s > 1e-4 * s[0]).sum())]
    sp = p[0] - p[0].mean(0)
    _, s2, vt2 = torch.linalg.svd(sp, full_matrices=False)
    k = int(np.searchsorted(np.cumsum((s2 ** 2).numpy()) / (s2 ** 2).sum().item(), 0.99)) + 1
    return time, vt2[:k]


def forward(model, sae, dm, pv, pe, cfg, x3_hook=None):
    """patch_embed + pe -> blocks 0..src_layer (x3) -> .. layer -> SAE codes z (2352, F) and the ordinary-token
    mask. x3_hook(x3) may return a grad-enabled replacement for x3."""
    v = model.vision.encoder
    x = v.patch_embed(pv.permute(0, 2, 1, 3, 4).float()) + pe
    for blk in v.blocks[: cfg["src_layer"] + 1]:
        x = blk(x)
    x3 = x if x3_hook is None else x3_hook(x)
    x = x3
    for blk in v.blocks[cfg["src_layer"] + 1: cfg["layer"] + 1]:
        x = blk(x)
    _, z = sae.encode(x[0] - dm)
    return x3, z, (x[0].norm(dim=-1) < cfg["massive_norm"]).float().detach()


def probe(model, sae, dm, pv, pe, cfg):
    """{cond: (12, n_feats) L5 profiles over ordinary tokens} under each PE condition."""
    fs, out = list(cfg["feats"]), {}
    with torch.no_grad():
        for cond in cfg["pe_conds"]:
            _, z, ok = forward(model, sae, dm, pv, pe_variant(pe, cond), cfg)
            out[cond] = (z[:, fs] * ok[:, None]).reshape(12, 196, -1).sum(1).numpy()
    return out


def l3_attr(model, sae, dm, pv, pe, bases, cfg):
    """Per feature: attribution x3.g summed over tokens, total and the parts in the PE time / space subspaces."""
    rows = []
    for f in cfg["feats"]:
        leaf = {}
        hook = lambda x: leaf.setdefault("x", x.detach().requires_grad_(True))
        x3, z, ok = forward(model, sae, dm, pv, pe, cfg, x3_hook=hook)
        (g,) = torch.autograd.grad((z[:, f] * ok).sum(), x3)
        x3, g = x3[0].detach(), g[0]
        row = {"feature": f, "total": float((x3 * g).sum())}
        for name, B in bases.items():
            row[name] = float(((x3 @ B.T) @ B * g).sum())
        rows.append(row)
    return rows


def main(cfg):
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    from umt_mlm_feature_contrast import clip_loader
    m = MODEL_REGISTRY["umt"]
    model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[("umt", "ssv2")]).eval()
    res = resolve_sae_checkpoint("umt", cfg["layer"], dataset_name="ssv2", sae_k=64, job_label="7ep")
    sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], "cpu", sae_k=64)
    dm, pe = torch.load(res["dim_mean_path"], weights_only=True).float(), model.vision.encoder.pos_embed.float()
    time, space = pe_bases(pe)
    rand = torch.linalg.qr(torch.randn(768, time.shape[0], generator=torch.Generator().manual_seed(0)))[0].T
    print(f"time basis {tuple(time.shape)}, space basis {tuple(space.shape)}, "
          f"time-space overlap (mean cos^2) {float(((time @ space.T) ** 2).sum(1).mean()):.3f}", flush=True)
    d = pd.read_parquet(cfg["dfa"], columns=["clip_id", "class_id"])
    d = pd.concat([d[d.class_id == c].head(cfg["max_clips"]) for c in cfg["classes"]]).reset_index(drop=True)
    rows = pd.DataFrame(dict(clip_id=d.clip_id.astype(str), class_id=d.class_id))
    prof, attr = [], []
    for pv, i in clip_loader(rows, cfg, "R"):
        i, cls = int(i), int(rows.class_id[int(i)])
        for cond, p in probe(model, sae, dm, pv, pe, cfg).items():
            prof += [dict(clip=rows.clip_id[i], cls=cls, pe=cond, feature=f, frame=t, z=float(p[t, j]))
                     for j, f in enumerate(cfg["feats"]) for t in range(12)]
        attr += [dict(r, clip=rows.clip_id[i], cls=cls)
                 for r in l3_attr(model, sae, dm, pv, pe, {"time": time, "space": space, "random11": rand}, cfg)]
        print(f"  {i + 1}/{len(rows)} {rows.clip_id[i]} ({cls})", flush=True)
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prof).to_parquet(cfg["out_dir"] / "profiles.parquet", index=False)
    pd.DataFrame(attr).to_parquet(cfg["out_dir"] / "l3_attr.parquet", index=False)
    print(f"Saved -> {cfg['out_dir']}", flush=True)


if __name__ == "__main__":
    main(CFG)
