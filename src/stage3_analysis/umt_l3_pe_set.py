"""UMT L3 pure-PE latent set (17/11/26). L3 SAE codes on a blank clip under full / no / reversed PE vs 8 real
down clips (43). PE-triggered = blank >= real, real > 1, <= 5% without PE, mirror under reversed PE >= 0.9; ramp = |r(frame)| >= .75.
Out: outputs/analysis/umt_l3_pe_lineage/l3_pe_set.csv (129 latents, 29 ramps). Usage: uv run python src/stage3_analysis/umt_l3_pe_set.py (repo root, CPU)."""
import sys, numpy as np, pandas as pd, torch
sys.path[:0] = ["notebooks", "src", "src/stage3_analysis"]
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
from umt_pe_probe import CFG as PCFG, inputs, pe_variant
from umt_l3_pe_lineage import CFG as LCFG
from umt_mlm_feature_contrast import clip_loader
L, m = 3, MODEL_REGISTRY["umt"]
model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[("umt", "ssv2")]).eval()
res = resolve_sae_checkpoint("umt", L, dataset_name="ssv2", sae_k=64, job_label="7ep")
sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], "cpu", sae_k=64)
dm, pe = torch.load(res["dim_mean_path"], weights_only=True).float(), model.vision.encoder.pos_embed.float()
def prof(pv, p):                                   # (12, F) L3 codes summed over each frame's tokens
    v = model.vision.encoder
    with torch.no_grad():
        x = v.patch_embed(pv.permute(0, 2, 1, 3, 4).float()) + p
        for blk in v.blocks[:L + 1]: x = blk(x)
        return sae.encode(x[0] - dm)[1].reshape(12, 196, -1).sum(1).numpy()
blank = inputs(PCFG)["blank"]; P = {c: prof(blank, pe_variant(pe, c)) for c in ("full", "none", "reversed")}
rows = pd.read_parquet(LCFG["dfa"], columns=["clip_id", "class_id"]).query("class_id == 43").head(8)
rows = pd.DataFrame(dict(clip_id=rows.clip_id.astype(str).values, class_id=rows.class_id.values))
real = np.mean([prof(pv, pe).sum(0) for pv, _ in clip_loader(rows, dict(LCFG, num_workers=0), "R")], 0)
a, b = P["full"] - P["full"].mean(0), P["reversed"][::-1] - P["reversed"][::-1].mean(0)
mir = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-9)
tf, tn, fr = P["full"].sum(0), P["none"].sum(0), np.arange(12.)
pe_set = np.where((tf >= real) & (real > 1) & (tn <= .05 * tf) & (mir >= .9))[0]
ramp = [np.corrcoef(P["full"][:, f], fr)[0, 1] if P["full"][:, f].std() > 0 else 0 for f in pe_set]
out = pd.DataFrame(dict(latent=pe_set, blank=tf[pe_set], real=real[pe_set], mirror=mir[pe_set], ramp_r=ramp))
out["ramp"] = out.ramp_r.abs() >= .75
out.to_csv("outputs/analysis/umt_l3_pe_lineage/l3_pe_set.csv", index=False)
print(len(out), "PE-triggered,", int(out.ramp.sum()), "ramps")
