"""Dictionary-wide PE characterisation of UMT SAE latents at L5 / L7 / L9 (19/11/26). One forward pass per input,
SAE codes read at every capture layer; per latent, per-frame activation summed over ordinary tokens.
Massive tokens: fixed norm >= 200 at every layer (the L5 cut; no clean gap at L7 / L9, so the excluded tokens are saved
as <key>_massive). Inputs: blank grey clip (full / frozen PE), the 34 single-object synthetic clips (full PE), 4 real val clips per class
(174 classes, R-correct DFA clip list) as R / C (shuffle) / Rev (frames flipped).
Out: outputs/analysis/umt_layer_pe_characterise/L{L}.npz (keys blank_full, blank_frozen (12, F); synth (34, 12, F);
R / C / Rev (clips, 12, F); n_massive_<key>) + clips.csv. GPU (Isambard).
Usage: [N_PER_CLASS=4 MAX_CLIPS=0] python -u src/stage3_analysis/umt_layer_pe_characterise.py"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "notebooks"), str(ROOT / "src"), str(ROOT / "src" / "stage3_analysis")]
from umt_pe_probe import CFG as PCFG, inputs, pe_variant
from umt_l3_pe_lineage import CFG as LCFG

CFG = dict(PCFG, layers=(5, 7, 9), n_per_class=int(os.environ.get("N_PER_CLASS", 4)),
           max_clips=int(os.environ.get("MAX_CLIPS", 0)), out=ROOT / "outputs/analysis/umt_layer_pe_characterise",
           clips=sorted(p.stem for p in (PCFG["clip_dir"] / "umt_12f").glob("*.pt")))


def load_all(cfg):
    """UMT SSv2 model + SAE and dim_mean for every capture layer (trained eval threshold via load_sae_eval)."""
    from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY, load_sae_eval, resolve_sae_checkpoint
    m = MODEL_REGISTRY["umt"]
    model = m["model_class"].from_pretrained(CHECKPOINT_REGISTRY[("umt", "ssv2")]).to(cfg["device"]).eval()
    saes = {}
    for L in cfg["layers"]:
        r = resolve_sae_checkpoint("umt", L, dataset_name="ssv2", sae_k=64, job_label="7ep")
        saes[L] = (load_sae_eval(r["sae_path"], m["hidden_dim"], m["num_patch_tokens"], cfg["device"], sae_k=64),
                   torch.load(r["dim_mean_path"], weights_only=True).to(cfg["device"]).float())
    return model, saes


def run_layers(model, saes, pv, pe, cfg):
    """patch_embed + pe -> blocks 0..max(L); at each capture layer: {L: (profile (12, F), n_massive (12,))}."""
    v, out = model.vision.encoder, {}
    with torch.no_grad():
        x = v.patch_embed(pv.permute(0, 2, 1, 3, 4).float().to(cfg["device"])) + pe.to(cfg["device"])
        for i, blk in enumerate(v.blocks[:max(cfg["layers"]) + 1]):
            x = blk(x)
            if i in saes:
                sae, dm = saes[i]
                ok = (x[0].norm(dim=-1) < cfg["massive_norm"]).float()           # same fixed cut at every layer
                _, z = sae.encode(x[0] - dm)
                prof = lambda w: (z * w[:, None]).reshape(12, 196, -1).sum(1).cpu().numpy().astype(np.float16)
                out[i] = (prof(ok), prof(1 - ok), (1 - ok).reshape(12, 196).sum(1).cpu().numpy())
    return out


def clip_rows(cfg):
    """First n_per_class clips of every class in the R-correct all-174 DFA clip list (MAX_CLIPS caps, for smoke)."""
    d = pd.read_parquet(LCFG["dfa"], columns=["clip_id", "class_id"]).sort_values(["class_id", "clip_id"])
    r = d.groupby("class_id").head(cfg["n_per_class"]).head(cfg["max_clips"] or None)
    return pd.DataFrame(dict(clip_id=r.clip_id.astype(str).values, class_id=r.class_id.values))


def main(cfg):
    from umt_mlm_feature_contrast import clip_loader
    model, saes = load_all(cfg)
    pe, acc = model.vision.encoder.pos_embed, {L: {} for L in cfg["layers"]}
    def add(key, out):                                             # append one input's profiles at every layer
        for L, (p, pm, nm) in out.items():
            acc[L].setdefault(key, []).append(p)
            acc[L].setdefault(f"{key}_massive", []).append(pm)          # excluded tokens: ordinary + massive = all
            acc[L].setdefault(f"n_massive_{key}", []).append(nm)
    ins = inputs(cfg)
    for c in ("full", "frozen"):
        add(f"blank_{c}", run_layers(model, saes, ins["blank"], pe_variant(pe, c), cfg))
    for n in cfg["clips"]:
        add("synth", run_layers(model, saes, ins[n], pe, cfg))
    rows = clip_rows(cfg)
    for cond, src in (("R", "R"), ("C", "C"), ("Rev", "R")):          # Rev = the R clip with frame order flipped
        for pv, i in clip_loader(rows, dict(LCFG, num_workers=cfg["num_workers"]), src):
            add(cond, run_layers(model, saes, pv.flip(1) if cond == "Rev" else pv, pe, cfg))
            for L in cfg["layers"]:                                    # row index actually loaded (decode retry
                acc[L].setdefault(f"row_{cond}", []).append(int(i))   # skips to the next clip on failure)
        print(f"{cond}: {len(rows)} clips done", flush=True)
    cfg["out"].mkdir(parents=True, exist_ok=True)
    for L, d in acc.items():
        np.savez(cfg["out"] / f"L{L}.npz", **{k: np.stack(v) for k, v in d.items()})
    rows.to_csv(cfg["out"] / "clips.csv", index=False)
    print("synth clips:", cfg["clips"], flush=True)


if __name__ == "__main__":
    CFG["num_workers"] = int(os.environ.get("NUM_WORKERS", 4))
    main(CFG)
