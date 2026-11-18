"""PE probe on real clips (18/11/26): L5 profiles (ordinary tokens) of 8 real down clips (43) under full / none /
reversed PE, every latent (CLS=<class> or NOUN=<object>). Feeds the feature sheet (umt_4250_feature_sheet.py).
Out: outputs/analysis/umt_pe_probe/real_{CLS or NOUN}.npz, key pe -> (clips, 12, F).
Usage: [CLS=43 | NOUN=ball] uv run python src/stage3_analysis/umt_pe_probe_real.py"""
import sys, numpy as np, pandas as pd, torch
sys.path[:0] = ["notebooks", "src", "src/stage3_analysis"]
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
from umt_pe_probe import CFG as PCFG, pe_variant, run_l5
from umt_l3_pe_lineage import CFG as LCFG
from umt_mlm_feature_contrast import clip_loader
import os
CFG = dict(PCFG, cls=int(os.environ.get("CLS", 43)), noun=os.environ.get("NOUN", ""), n_clips=8,
           pe_conds=["full", "none", "reversed"],
           obj_sets=PCFG["out_dir"].parent / "umt_object_contrast/sets.csv")   # NOUN: first n 'pos' clips of that object
CFG["tag"] = CFG["noun"] or str(CFG["cls"])                                   # -> real_{tag}.npz (real_43 = #4250 run)


def clip_rows(cfg):
    """(clip_id, class_id) of the real clips: the class's first n DFA clips, or the noun's first n 'pos' clips."""
    if cfg["noun"]:
        s = pd.read_csv(cfg["obj_sets"], dtype={"clip_id": str}).query(f"object == '{cfg['noun']}' and role == 'pos'")
        return pd.DataFrame(dict(clip_id=s.clip_id.values, class_id=s.true_class.values)).head(cfg["n_clips"])
    r = pd.read_parquet(LCFG["dfa"], columns=["clip_id", "class_id"]).query(f"class_id == {cfg['cls']}").head(cfg["n_clips"])
    return pd.DataFrame(dict(clip_id=r.clip_id.astype(str).values, class_id=r.class_id.values))


def main(cfg):
    m = MODEL_REGISTRY["umt"]
    model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[("umt", "ssv2")]).to(cfg["device"]).eval()
    res = resolve_sae_checkpoint("umt", 5, dataset_name="ssv2", sae_k=64, job_label="7ep")
    sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=64)
    dm = torch.load(res["dim_mean_path"], weights_only=True).to(cfg["device"]).float()
    rows = clip_rows(cfg)
    pe, out = model.vision.encoder.pos_embed, {c: [] for c in cfg["pe_conds"]}
    for pv, _ in clip_loader(rows, dict(LCFG, num_workers=0), "R"):
        for c in cfg["pe_conds"]:
            out[c].append(run_l5(model, sae, dm, pv, pe_variant(pe, c), cfg)[0].astype(np.float16))
    np.savez(cfg["out_dir"] / f"real_{cfg['tag']}.npz", **{c: np.stack(v) for c, v in out.items()})


if __name__ == "__main__":
    main(CFG)
