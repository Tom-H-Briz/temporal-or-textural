"""R / C (shuffle) / Rev L5 profiles for a few classes (18/11/26; first use: #4638, #186 on up/down + camera up/down).
Ordinary tokens, full PE, every latent. Out: outputs/analysis/umt_feature_rc_rev/profiles.npz (cond -> (clips, 12, F))
+ clips.csv. Usage: [CLASSES=43,45,168,165 N_CLIPS=12] uv run python src/stage3_analysis/umt_feature_rc_rev.py"""
import os, sys, numpy as np, pandas as pd, torch
sys.path[:0] = ["notebooks", "src", "src/stage3_analysis"]
from umt_pe_probe_real import CFG as PRCFG                 # same model / SAE / token settings
from umt_pe_probe import run_l5
from umt_l3_pe_lineage import CFG as LCFG
from umt_mlm_feature_contrast import clip_loader
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
CFG = dict(PRCFG, classes=[int(c) for c in os.environ.get("CLASSES", "43,45,168,165").split(",")],
           n_clips=int(os.environ.get("N_CLIPS", 12)), out=PRCFG["out_dir"].parent / "umt_feature_rc_rev")


def main(cfg):
    m = MODEL_REGISTRY["umt"]
    model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[("umt", "ssv2")]).to(cfg["device"]).eval()
    res = resolve_sae_checkpoint("umt", 5, dataset_name="ssv2", sae_k=64, job_label="7ep")
    sae = load_sae_eval(res["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=64)
    dm, pe = torch.load(res["dim_mean_path"], weights_only=True).to(cfg["device"]).float(), model.vision.encoder.pos_embed
    d = pd.read_parquet(LCFG["dfa"], columns=["clip_id", "class_id"])
    rows = pd.concat([d[d.class_id == c].head(cfg["n_clips"]) for c in cfg["classes"]])
    rows = pd.DataFrame(dict(clip_id=rows.clip_id.astype(str).values, class_id=rows.class_id.values))
    out = {}
    for cond, src in (("R", "R"), ("C", "C"), ("Rev", "R")):          # Rev = R clip with frame order flipped
        out[cond] = np.stack([run_l5(model, sae, dm, pv.flip(1) if cond == "Rev" else pv, pe, cfg)[0].astype(np.float16)
                              for pv, _ in clip_loader(rows, dict(LCFG, num_workers=0), src)])
    cfg["out"].mkdir(exist_ok=True)
    np.savez(cfg["out"] / "profiles.npz", **out)
    rows.to_csv(cfg["out"] / "clips.csv", index=False)


if __name__ == "__main__":
    main(CFG)
