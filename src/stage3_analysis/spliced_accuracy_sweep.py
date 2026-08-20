"""
Comprehensive spliced-accuracy sweep — every current (post-30/07-fix) VideoMAE
SAE, one canonical held-out clip set per dataset, one run.

Scope: VideoMAE only (SSv2 + K400, job7ep, k64/x8 and k128/x16). TimeSformer is
excluded — spliced_accuracy_tf.py has no reusable function (everything is inline
in main(), driven by its own module-level CFG), so forcing it through this VM-
shaped pipeline risks silently wrong num_frames/cls_offset. Needs a small
spliced_accuracy_tf.py refactor first; out of scope here.

Canonical eval set, both datasets, split_seed=42:
  - SSv2: no held-out list was ever persisted (train_sae.py only persists one in
    its val_fraction branch, which SSv2 never uses — see build_split there). This
    script regenerates it via train_sae.py's own build_split()/persist_held_out_
    clips(), so it is byte-identical to what every real SSv2 job7ep training run
    already implicitly used for R2/dead-feature selection — not a new split.
  - K400: reuses the already-persisted outputs/sae/videomae_kinetics400_held_out_
    val_clips.json (3,976 clips) — no need to regenerate.

Outputs (outputs/analysis/spliced_accuracy_sweep/):
    spliced_accuracy_sweep_per_clip.parquet   — one row per (config, clip)
    spliced_accuracy_sweep_summary.csv        — one row per config, overall numbers
    {config}.csv                              — per-class + OVERALL, one per config
                                                 (written by run_spliced_accuracy itself)

Usage:
    uv run python src/stage3_analysis/spliced_accuracy_sweep.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "stage2_sae"))
sys.path.insert(0, str(ROOT / "notebooks"))

from train_sae import CFG as TRAIN_CFG, build_split, persist_held_out_clips
from ToT_utils import resolve_sae_checkpoint
from spliced_accuracy_vm import run_spliced_accuracy, CFG as SPLICE_CFG

CFG = {
    "out_dir": ROOT / "outputs" / "analysis" / "spliced_accuracy_sweep",
    "ssv2_held_out_path": ROOT / "outputs" / "sae" / "videomae_ssv2_held_out_val_clips.json",
    "k400_held_out_path": ROOT / "outputs" / "sae" / "videomae_kinetics400_held_out_val_clips.json",
    # run_spliced_accuracy reads a single VIDEO_DIR env var regardless of dataset —
    # fine for every existing single-dataset SLURM job, but this script processes
    # both datasets in one process, so VIDEO_DIR must be set fresh per call rather
    # than once for the whole job (a single export would point K400 lookups at the
    # SSv2 directory or vice versa). SSV2_VIDEO_DIR/K400_VIDEO_DIR let the SLURM
    # launcher set both explicitly; unset falls back to DATASET_REGISTRY's defaults.
    "video_dir_by_dataset": {
        "ssv2":        os.environ.get("SSV2_VIDEO_DIR"),
        "kinetics400": os.environ.get("K400_VIDEO_DIR"),
    },
}

CONFIGS = [
    {"layer": l, "sae_k": k, "dataset": "ssv2"}
    for l in (3, 5, 7, 9) for k in (64, 128) if not (k == 128 and l == 3)
] + [
    {"layer": l, "sae_k": k, "dataset": "kinetics400"}
    for l in (3, 5, 7, 9) for k in (64, 128) if not (k == 128 and l == 3)
]


def ensure_ssv2_held_out() -> list[str]:
    """Regenerate + persist the SSv2 held-out val list if it doesn't exist yet, via
    train_sae.py's own build_split()/persist_held_out_clips() — not reimplemented,
    so this is guaranteed byte-identical to what real SSv2 training runs used."""
    if CFG["ssv2_held_out_path"].exists():
        return json.load(open(CFG["ssv2_held_out_path"]))
    assert TRAIN_CFG["model_name"] == "videomae" and TRAIN_CFG["dataset_name"] == "ssv2", (
        "train_sae.py's default CFG changed — this script assumes its defaults "
        "match a real SSv2 job7ep training run; re-check before regenerating."
    )
    _, val_paths = build_split(TRAIN_CFG)
    persist_held_out_clips(val_paths, TRAIN_CFG)
    return [p.name for p in val_paths]


def run_one_config(cfg: dict, eval_clips: dict[str, list[str]]) -> tuple[dict, pd.DataFrame]:
    """One SAE, spliced accuracy on its dataset's canonical held-out set. Returns
    (summary row, per-clip rows) — the per-class CSV is written as a side effect
    by run_spliced_accuracy itself, redirected into this sweep's own out_dir."""
    layer, sae_k, dataset = cfg["layer"], cfg["sae_k"], cfg["dataset"]
    name = f"VM_{dataset}_L{layer}_k{sae_k}"
    resolved = resolve_sae_checkpoint("videomae", layer, dataset_name=dataset, sae_k=sae_k)
    print(f"\n=== {name} === {resolved['sae_path']}")

    override = CFG["video_dir_by_dataset"][dataset]
    if override:
        os.environ["VIDEO_DIR"] = override
    else:
        os.environ.pop("VIDEO_DIR", None)  # fall back to DATASET_REGISTRY's default

    splice_cfg = dict(SPLICE_CFG, output_dir=str(CFG["out_dir"]))
    result = run_spliced_accuracy(
        sae_checkpoint=resolved["sae_path"], layer=layer, model_name="videomae",
        dataset_name=dataset, eval_clips=eval_clips[dataset],
        dim_mean_path=resolved["dim_mean_path"], return_per_clip=True, cfg=splice_cfg,
    )
    per_clip = result.pop("per_clip_df")
    per_clip = per_clip.assign(config=name, dataset=dataset, layer=layer, sae_k=sae_k)
    per_clip = per_clip[["config", "dataset", "layer", "sae_k", "clip_id", "class_id",
                          "baseline_correct", "spliced_correct"]]
    summary = {"config": name, **cfg, **{k: v for k, v in result.items() if k != "csv_path"}}
    return summary, per_clip


def main() -> None:
    CFG["out_dir"].mkdir(parents=True, exist_ok=True)
    eval_clips = {
        "ssv2": ensure_ssv2_held_out(),
        "kinetics400": json.load(open(CFG["k400_held_out_path"])),
    }
    print(f"Eval pool sizes: ssv2={len(eval_clips['ssv2'])}  kinetics400={len(eval_clips['kinetics400'])}")

    summaries, per_clip_frames = [], []
    for cfg in CONFIGS:
        summary, per_clip = run_one_config(cfg, eval_clips)
        summaries.append(summary)
        per_clip_frames.append(per_clip)

    pd.concat(per_clip_frames, ignore_index=True).to_parquet(
        CFG["out_dir"] / "spliced_accuracy_sweep_per_clip.parquet"
    )
    pd.DataFrame(summaries).to_csv(CFG["out_dir"] / "spliced_accuracy_sweep_summary.csv", index=False)
    print(f"\nDone — {len(CONFIGS)} configs -> {CFG['out_dir']}")


if __name__ == "__main__":
    main()
