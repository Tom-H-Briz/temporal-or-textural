"""TimeSformer blank-only PE split (24/11/26; findings/pe_timing_frozen_real_three_backbones_findings_241126.md §2).
TF keeps space and time in separate PE tables, so on the grey blank each can be removed on its own: which table drives
each latent? Variants: time table zeroed, spatial PE zeroed (CLS row kept), every patch = patch 0's spatial row, both
zeroed. Only TF: UMT's PE is one 1D sinusoid over the token index and VM's is joint, so there the frozen test is the
only lever. CPU, local, no videos. Out: outputs/analysis/tf_layer_pe_characterise/blank_pe_split_L{L}.csv (per latent,
blank activity per variant) + blank_pe_split_summary.csv (median kept vs full blank, active pure PE / mixed).
Usage: uv run python src/stage3_analysis/tf_blank_pe_split.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src" / "stage3_analysis")]
import tf_layer_pe_characterise as tpc

CFG = dict(tpc.CFG, device="cpu", active=0.5, groups=("pe", "mixed"),
           variants=("full", "time_zero", "space_zero", "space_frozen", "both_zero"))


def pe_tables(sp, te, variant):
    """(spatial (1, 197, D), time (1, 8, D)) for one variant. Row 0 of the spatial table is CLS's, always kept."""
    zero_patches = torch.cat([sp[:, :1], torch.zeros_like(sp[:, 1:])], 1)
    same_patch = torch.cat([sp[:, :1], sp[:, 1:2].expand(-1, sp.shape[1] - 1, -1)], 1)
    return {"full": (sp, te), "time_zero": (sp, 0 * te), "space_zero": (zero_patches, te),
            "space_frozen": (same_patch, te), "both_zero": (zero_patches, 0 * te)}[variant]


def blank_by_variant(model, proc, saes, cfg):
    """{variant: {L: clip-summed blank activity (F,)}}. The spatial table is swapped in place and always restored."""
    emb, out = model.timesformer.embeddings, {}
    sp0, te0 = emb.position_embeddings.data.clone(), emb.time_embeddings.data.clone()
    try:
        for v in cfg["variants"]:
            emb.position_embeddings.data, te = pe_tables(sp0, te0, v)
            res = tpc.run_layers(model, saes, tpc.blank(cfg, proc), te, cfg)
            out[v] = {L: r[0].astype(np.float64).sum(0) for L, r in res.items()}
    finally:
        emb.position_embeddings.data = sp0
    return out


def tables(out, cfg):
    """Per layer: per-latent csv (group, fire rate, blank activity per variant); summary of median kept vs full."""
    rows = []
    for L in cfg["layers"]:
        f = pd.read_csv(cfg["out"] / f"features_L{L}.csv")
        d = f[["latent", "cls", "fire_rate"]].assign(**{v: out[v][L] for v in cfg["variants"]})
        d.to_csv(cfg["out"] / f"blank_pe_split_L{L}.csv", index=False)
        for g in cfg["groups"]:
            a = d[(d.cls == g) & (d.fire_rate >= cfg["active"]) & (d.full > 0)]
            rows.append(dict(layer=L, group=g, n=len(a), **{f"kept_{v}": np.median(a[v] / a.full) for v in cfg["variants"][1:]}))
    return pd.DataFrame(rows)


def main(cfg):
    model, proc, saes = tpc.load_all(cfg)
    s = tables(blank_by_variant(model, proc, saes, cfg), cfg)
    s.round(4).to_csv(cfg["out"] / "blank_pe_split_summary.csv", index=False)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main(CFG)
