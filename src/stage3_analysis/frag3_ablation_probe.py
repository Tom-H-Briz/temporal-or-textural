"""
SAE-ablation probe of the slot-6 fragment group {2179, 935, 5721}.

Hypothesis under test: the fragments — the largest non-scaffold contributors to
the scaffold subspace (in-span fractions 0.36/0.29/0.26) — causally contribute
to the toward/away class readout. SAE ablation, deliberately: zero the three
latents in the spliced reconstruction (the run_ablation.py operation), so the
result lands in the same substrate as every existing ablation number and joins
them directly. Raw-space translation is out of scope (workbook 220926: the
decode map is many-to-one; that question is declared endless).

Decked population: 50 clips each of class 42 (closer) / 40 (away) + 300
population-stratified comparator excluding all six toward/away classes. R-correct
filter, matching the SAE line's convention. Head-to-head: existing all7
per-clip deltas (condition R) for classes 40/42 from the same source parquet.

Usage:
    uv run python src/stage3_analysis/frag3_ablation_probe.py --n-clips 4
    uv run python src/stage3_analysis/frag3_ablation_probe.py
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

from stage3_analysis.dfa_engine import _preprocess_clip
from stage3_analysis.l5_ablation_l7_feature_impact import load_sae
from stage3_analysis.raw_stream_feature_ablation import make_l5_hook
from ToT_utils import CHECKPOINT_REGISTRY, MODEL_REGISTRY

CFG = {
    "model_flag": "videomae",
    "dataset": "ssv2",
    "layer": 5,
    "sae_k": 64,
    "frag3": [2179, 935, 5721],       # slot-6 fragments, in-span .363/.289/.256
    "deck_classes": {42: "closer", 40: "away"},
    "excluded_from_other": [40, 42, 32, 36, 37, 41],   # all toward/away classes
    "n_deck_per_class": 50,
    "n_other": 300,
    "seed": 0,
    "checkpoint_every": 25,
    "device": "cuda" if torch.cuda.is_available()
              else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "source_parquet": ROOT / "outputs/analysis/scaffold_ablation/"
                                 "ablation_results_long_l5_job7ep_k64.parquet",
    "video_dir": ROOT / "data/ssv2/20bn-something-something-v2",
    "out_dir": ROOT / "outputs/analysis/frag3_probe",
}


def sample_population(cfg: dict) -> pd.DataFrame:
    """Decked (closer/away) + population-stratified comparator, from the
    R-correct parquet — the SAE line's own population."""
    df = pd.read_parquet(cfg["source_parquet"],
                         columns=["clip_id", "class_id", "sl_label"]).drop_duplicates("clip_id")
    df = df[df.clip_id.astype(str).str.removesuffix(".webm").apply(
        lambda c: (cfg["video_dir"] / f"{c}.webm").exists())]
    df["clip_id"] = df.clip_id.astype(str).str.removesuffix(".webm")
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for cid, tag in cfg["deck_classes"].items():
        pool_ = df[df.class_id == cid]
        rows.append(pool_.iloc[rng.choice(len(pool_), size=cfg["n_deck_per_class"],
                                          replace=False)].assign(group=tag))
    rest = df[~df.class_id.isin(cfg["excluded_from_other"])]
    quota = (rest.sl_label.value_counts(normalize=True) * cfg["n_other"]).round().astype(int)
    for label, n in quota.items():
        pool_ = rest[rest.sl_label == label]
        rows.append(pool_.iloc[rng.choice(len(pool_), size=int(n), replace=False)]
                    .assign(group="other"))
    return pd.concat(rows).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-clips", type=int, default=0, help="smoke-test cutoff")
    args = parser.parse_args()
    cfg = CFG
    cfg["out_dir"].mkdir(parents=True, exist_ok=True)

    model_cfg = MODEL_REGISTRY[cfg["model_flag"]]
    checkpoint = CHECKPOINT_REGISTRY[(cfg["model_flag"], cfg["dataset"])]
    processor = model_cfg["processor_class"].from_pretrained(checkpoint)
    model = model_cfg["model_class"].from_pretrained(checkpoint).to(cfg["device"]).eval()
    model.requires_grad_(False)
    sae, dim_mean = load_sae(cfg["model_flag"], cfg["layer"], cfg["sae_k"], cfg["device"])
    dictionary = sae.dictionary.get_dictionary().detach()

    state, capture = {"mode": "recon", "indices": [], "dirs": None}, {}
    model_cfg["layer_getter"](model, cfg["layer"]).register_forward_hook(
        make_l5_hook(sae, dim_mean, dictionary, model_cfg["cls_offset"], state, capture))

    clips = sample_population(cfg)
    if args.n_clips > 0:
        clips = clips.head(args.n_clips)
    out_path = cfg["out_dir"] / "frag3_probe_results.parquet"
    done = set(pd.read_parquet(out_path).clip_id) if out_path.exists() else set()
    todo = clips[~clips.clip_id.isin(done)]
    print(f"Device: {cfg['device']}  {len(todo)} clips x 2 passes "
          f"({len(done)} already done)")

    rows, t0 = [], time.time()
    for i, (_, r) in enumerate(todo.iterrows()):
        pv = _preprocess_clip(cfg["video_dir"] / f"{r.clip_id}.webm",
                              model_cfg["num_frames"], processor, cfg["device"])
        for mode, idx in [("recon", []), ("recon_minus", cfg["frag3"])]:
            state["mode"], state["indices"] = mode, idx
            with torch.no_grad():
                logits = model(pixel_values=pv).logits.squeeze(0)
            rows.append({"clip_id": r.clip_id, "class_id": int(r.class_id),
                         "group": r.group, "arm": mode,
                         "logit": float(logits[int(r.class_id)]),
                         "pred": int(logits.argmax()),
                         "correct": int(logits.argmax()) == int(r.class_id)})
        if (i + 1) % cfg["checkpoint_every"] == 0 or i == len(todo) - 1:
            merged = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
            pd.concat([merged, pd.DataFrame(rows)]).drop_duplicates(
                ["clip_id", "arm"]).to_parquet(out_path, index=False)
            rows = []
            print(f"  [{i+1}/{len(todo)}]  {(time.time()-t0)/(i+1):.1f}s/clip")
    summarise(pd.read_parquet(out_path), cfg)


def summarise(df: pd.DataFrame, cfg: dict) -> None:
    base = df[df.arm == "recon"].set_index("clip_id")
    abl = df[df.arm == "recon_minus"]
    abl = abl.assign(base_logit=abl.clip_id.map(base.logit),
                     base_correct=abl.clip_id.map(base.correct))
    abl["delta"] = abl.base_logit - abl.logit
    abl["flip"] = abl.base_correct & ~abl.correct
    out = abl.groupby("group").agg(
        n=("delta", "size"), mean_delta=("delta", "mean"),
        flip_rate=("flip", "mean")).round(4).reset_index()
    src = pd.read_parquet(cfg["source_parquet"],
                          columns=["clip_id", "class_id", "ablation_target",
                                   "perturbation_condition", "delta"])
    all7 = src[(src.ablation_target == "all7") & (src.perturbation_condition == "R")]
    for cid, tag in cfg["deck_classes"].items():
        ref = all7[all7.class_id == cid].delta
        out.loc[out.group == tag, "all7_ref_delta"] = round(float(ref.mean()), 4)
        out.loc[out.group == tag, "all7_ref_n"] = len(ref)
    out.to_csv(cfg["out_dir"] / "frag3_probe_summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"-> {cfg['out_dir']}")


if __name__ == "__main__":
    main()
