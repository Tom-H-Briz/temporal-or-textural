"""
Raw-stream scaffold dose-response: scale the 7 L5 scaffold contributions on
the model's OWN activations (error term intact).

    h' = h_raw - (1 - alpha) * sum_i z_i d_i        (i in all7)

alpha=1 is the untouched raw pass; alpha=0 is exactly raw_minus (subtraction,
the n4450 bracket's lower bound); alpha=2 is the BOOST — the features'
attributed mark added on top of the real activation. This is the cleaner
"would more signal help" test: substrate is the real activation, only the
mark is scaled, no splice anywhere.

Population: the same manifest ranks 0-499 as the splice-space dose-response
(identical clips, direct comparison). Representative (not R-correct), ~81%
raw-correct, so improvement is measurable. Nothing raw-space is on file for
these clips — all arms in-session.

Pre-registered reading: raw x2 temporal gain with CI excluding 0 -> real
utilisation headroom the splice masked; flat -> maximally utilised holds on
the real substrate.

Usage:
    uv run python src/stage3_analysis/scaffold_dose_raw.py --n-clips 4
    uv run python src/stage3_analysis/scaffold_dose_raw.py
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
    "alphas": [1.0, 0.0, 2.0],
    "n_clips": 500,
    "checkpoint_every": 25,
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "manifest": ROOT / "outputs/analysis/pos_embed_shuffle/clip_order_manifest.csv",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "out_dir": ROOT / "outputs/analysis/scaffold_dose_raw",
}


def make_raw_scale_hook(sae, dim_mean, dictionary, features: list[int],
                        cls_offset: int, state: dict):
    """L5 hook: h' = flat - (1 - alpha) * (z[:, feats] @ dictionary[feats]).
    alpha=1 returns None (untouched); alpha=0 is raw_minus exactly."""
    def hook_fn(module, input, output):
        if state["alpha"] == 1.0:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        cls, patches = hidden[:, :cls_offset], hidden[:, cls_offset:]
        B, T, D = patches.shape
        flat = patches.reshape(B * T, D).float()
        with torch.no_grad():
            _, z = sae.encode(flat - dim_mean)
            new = flat - (1.0 - state["alpha"]) * (z[:, features] @ dictionary[features])
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
    dictionary = sae.dictionary.get_dictionary().detach()

    state = {"alpha": 1.0}
    model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(
        make_raw_scale_hook(sae, dim_mean, dictionary, features,
                            model_cfg["cls_offset"], state))

    m = pd.read_csv(cfg["manifest"], dtype={"clip_id": str})
    clips = m[m["rank"] < args.n_clips]
    out_path = cfg["out_dir"] / "scaffold_dose_raw_results.parquet"
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
    b = df[df.alpha == 1.0].set_index("clip_id")
    d = df[df.alpha != 1.0].assign(
        base_logit=b.logit.reindex(df[df.alpha != 1.0].clip_id).values,
        base_correct=b.correct.reindex(df[df.alpha != 1.0].clip_id).values)
    d["delta"] = d.base_logit - d.logit
    out = []
    for alpha, g in d.groupby("alpha"):
        for scope, sub in [("overall", g)] + [(l, g[g.sl_label == l])
                                              for l in ("temporal", "static")]:
            v = sub.delta
            se = v.std(ddof=1) / np.sqrt(len(v))
            out.append({"alpha": alpha, "scope": scope, "n": len(sub),
                        "acc_base": sub.base_correct.mean(), "acc_alpha": sub.correct.mean(),
                        "mean_delta": v.mean(), "ci95_lo": v.mean() - 1.96 * se,
                        "ci95_hi": v.mean() + 1.96 * se,
                        "improve_rate": (~sub.base_correct & sub.correct).mean(),
                        "damage_rate": (sub.base_correct & ~sub.correct).mean()})
    s = pd.DataFrame(out)
    s.to_csv(cfg["out_dir"] / "scaffold_dose_raw_summary.csv", index=False)
    print(s.to_string(index=False))
    print("\nreferences: n4450 raw_minus x0 = 0.249 | splice-space x2 = +0.023 n.s.")
    print("boost reading: temporal x2 delta CI excluding 0 -> headroom; else flat")


if __name__ == "__main__":
    main()
