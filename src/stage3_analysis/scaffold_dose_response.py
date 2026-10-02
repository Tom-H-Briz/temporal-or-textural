"""
Scaffold dose-response: scale the 7 L5 scaffold latents and measure behaviour.

Arms (alpha in {1, 0, 0.5, 2, 4}): z[:, all7] *= alpha -> decode -> splice.
alpha=0 is exactly recon_minus (the on-file ablation operation); alpha>1 asks
whether MORE scaffold signal IMPROVES temporal classes. Population is the
pos-embed manifest's first 500 ranks — representative (not R-correct), so
clips wrong at base can get better; nothing on file, everything in-session.

Questions pre-registered:
  - does x0.5 sit between x0 and x1 (linear chord) or threshold-like?
  - does x2/x4 lift temporal-class accuracy above the x1 spliced baseline?
Caveat pre-registered: x4 is off-manifold by construction; a non-monotone or
saturating upper arm is a finding, not a failure.

Usage:
    uv run python src/stage3_analysis/scaffold_dose_response.py --n-clips 4
    uv run python src/stage3_analysis/scaffold_dose_response.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from stage3_analysis.ablation_targets import get_targets
from stage3_analysis.dfa_engine import _preprocess_clip
from stage3_analysis.l5_ablation_l7_feature_impact import load_sae
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "layer": 5,
    "sae_k": 64,
    "alphas": [1.0, 0.0, 0.5, 2.0, 4.0],
    "n_clips": 500,
    "checkpoint_every": 25,
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "manifest": ROOT / "outputs/analysis/pos_embed_shuffle/clip_order_manifest.csv",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "out_dir": ROOT / "outputs/analysis/scaffold_dose_response",
}


def make_scale_hook(sae, dim_mean, features: list[int], cls_offset: int, state: dict):
    """L5 splice hook with scaffold latents scaled by state['alpha']. At
    alpha=1 this is the plain splice; at 0 it is recon_minus."""
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        cls, patches = hidden[:, :cls_offset], hidden[:, cls_offset:]
        B, T, D = patches.shape
        flat = patches.reshape(B * T, D).float()
        with torch.no_grad():
            _, z = sae.encode(flat - dim_mean)
            z_scaled = z.clone()
            z_scaled[:, features] = z_scaled[:, features] * state["alpha"]
            new = sae.decode(z_scaled) + dim_mean
        new = new.to(hidden.dtype).reshape(B, T, D)
        out = torch.cat([cls, new], dim=1) if cls_offset else new
        return (out,) + output[1:] if isinstance(output, tuple) else out
    return hook_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=CFG["n_clips"])
    args = parser.parse_args()
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    features = get_targets(cfg["dataset"], cfg["layer"])["all7"]
    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], cfg["dataset"])]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(cfg["device"]).eval()
    model.requires_grad_(False)
    sae, dim_mean = load_sae(cfg["model_flag"], cfg["layer"], cfg["sae_k"], cfg["device"])

    state = {"alpha": 1.0}
    model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(
        make_scale_hook(sae, dim_mean, features, model_cfg["cls_offset"], state))

    m = pd.read_csv(cfg["manifest"], dtype={"clip_id": str})
    clips = m[m["rank"] < args.n_clips]
    out_path = cfg["out_dir"] / "scaffold_dose_results.parquet"
    done = set(zip(*[pd.read_parquet(out_path, columns=["clip_id", "alpha"])[c]
                     for c in ("clip_id", "alpha")])) if out_path.exists() else set()
    todo = clips[~clips.apply(
        lambda r: all((r.clip_id, a) in done for a in cfg["alphas"]), axis=1)]
    print(f"Device: {cfg['device']}  features {features}  "
          f"{len(todo)} clips x {len(cfg['alphas'])} passes")

    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(todo.iterrows()):
        pv = _preprocess_clip(cfg["video_dir"] / f"{r.clip_id}.webm",
                              model_cfg["num_frames"], processor, cfg["device"])
        for alpha in cfg["alphas"]:
            if (r.clip_id, alpha) in done:
                continue
            state["alpha"] = alpha
            with torch.no_grad():
                logits = model(pixel_values=pv).logits.squeeze(0)
            rows.append({"clip_id": r.clip_id, "class_id": int(r.class_id),
                         "sl_label": r.sl_label, "alpha": alpha,
                         "logit": float(logits[int(r.class_id)]),
                         "pred": int(logits.argmax()),
                         "correct": int(logits.argmax()) == int(r.class_id)})
        if (i + 1) % cfg["checkpoint_every"] == 0 or i == len(todo) - 1:
            merged = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
            pd.concat([merged, pd.DataFrame(rows)]).drop_duplicates(
                ["clip_id", "alpha"]).to_parquet(out_path, index=False)
            rows = []
            print(f"  [{i+1}/{len(todo)}]  {(time.time()-t0)/(i+1):.1f}s/clip")
    summarise(pd.read_parquet(out_path), cfg)


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    base = df[df.alpha == 1.0].set_index("clip_id")
    d = df[df.alpha != 1.0].assign(
        base_logit=df[df.alpha == 1.0].set_index("clip_id").logit.reindex(
            df[df.alpha != 1.0].clip_id).values,
        base_correct=df[df.alpha == 1.0].set_index("clip_id").correct.reindex(
            df[df.alpha != 1.0].clip_id).values)
    d["delta"] = d.base_logit - d.logit
    d["improved"] = ~d.base_correct & d.correct
    out = []
    for alpha, g in d.groupby("alpha"):
        for scope, sub in [("overall", g)] + [(l, g[g.sl_label == l])
                                              for l in ("temporal", "static")]:
            v = sub.delta
            se = v.std(ddof=1) / np.sqrt(len(v))
            out.append({"alpha": alpha, "scope": scope, "n": len(sub),
                        "acc_base": sub.base_correct.mean(),
                        "acc_alpha": sub.correct.mean(),
                        "mean_delta": v.mean(),
                        "ci95_lo": v.mean() - 1.96 * se, "ci95_hi": v.mean() + 1.96 * se,
                        "improve_rate": sub.improved.mean(),
                        "damage_rate": (sub.base_correct & ~sub.correct).mean()})
    s = pd.DataFrame(out)
    s.to_csv(cfg["out_dir"] / "scaffold_dose_summary.csv", index=False)
    print(s.to_string(index=False))

    for scope in ("overall", "temporal", "static"):
        g = s[(s.scope == scope)].set_index("alpha")
        d0, dh = g.loc[0.0, "mean_delta"], g.loc[0.5, "mean_delta"]
        chord = 0.5 * d0
        print(f"\n[{scope}] x0.5 delta {dh:+.4f} vs linear chord {chord:+.4f} "
              f"(ratio {dh / chord:.2f}; <1 sub-linear, >1 super-linear)")
        for a in (2.0, 4.0):
            print(f"[{scope}] x{a:g}: acc {g.loc[a, 'acc_base']:.3f} -> "
                  f"{g.loc[a, 'acc_alpha']:.3f}  (improve rate "
                  f"{g.loc[a, 'improve_rate']:.3f})")


if __name__ == "__main__":
    main()
