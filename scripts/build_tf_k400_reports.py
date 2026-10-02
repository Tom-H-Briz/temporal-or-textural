"""
Assemble the TF-K400 run reports after the chain completes.

Writes (all under outputs/tf_k400/):
  verification.md  — file counts per stage (expected 3/3/3/3), exit non-zero on shortfall
  sae_health.md    — per-layer R^2 / L0 / dead counts from the train_sae .out logs,
                     plus k / nb_concepts read back from the checkpoints themselves
  spliced_accuracy.md — baseline vs spliced vs drop per layer, from the Job 3 CSVs
  dfa_summary.md   — per-layer DFA parquet row counts + four-bucket pool sizes

Numbers as found, no interpretation. Health metrics' canonical home is wandb —
this is the local collation of the same printed values.

Usage (inside the final chain job): python scripts/build_tf_k400_reports.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs" / "tf_k400"
LAYERS = (5, 7, 9)

CFG = {
    "sae_dir":         ROOT / "outputs" / "sae",
    "spliced_dir":     ROOT / "outputs" / "spliced_accuracy_vm",
    "dfa_dir":         ROOT / "outputs" / "analysis" / "dfa_mass_delta_tf",
    "bucket_dir":      ROOT / "outputs" / "analysis" / "shuffle_reduction_composition",
    "train_out_glob":  "tf_k400_train_sae_*.out",
    "spliced_pattern": "spliced_accuracy_l{layer}_kinetics400_sae_tf_kinetics400_k64_x8_l{layer}_job7ep_best.csv",
    "sae_pattern":     "sae_tf_kinetics400_k64_x8_l{layer}_job7ep_best.pt",
    "dim_mean":        "tf_kinetics400_layer{layer}_dim_mean.pt",
    "dfa_pattern":     "dfa_mass_delta_tf_kinetics400_l{layer}_job7ep_k64.parquet",
    "bucket_pattern":  "k400_tf_l{layer}_clip_shuffle_disruption.csv",
}


def verify_counts() -> dict[str, int]:
    counts = {
        "dim_mean": sum((CFG["sae_dir"] / CFG["dim_mean"].format(layer=l)).exists() for l in LAYERS),
        "sae":      sum((CFG["sae_dir"] / CFG["sae_pattern"].format(layer=l)).exists() for l in LAYERS),
        "spliced":  sum((CFG["spliced_dir"] / CFG["spliced_pattern"].format(layer=l)).exists() for l in LAYERS),
        "dfa":      sum((CFG["dfa_dir"] / CFG["dfa_pattern"].format(layer=l)).exists() for l in LAYERS),
    }
    lines = ["# Verification counts (expected 3/3/3/3)", ""]
    lines += [f"- {k}: {v}/3" for k, v in counts.items()]
    (OUT / "verification.md").write_text("\n".join(lines) + "\n")
    if any(v != 3 for v in counts.values()):
        raise SystemExit(f"VERIFICATION FAILED: {counts} — expected 3/3/3/3")
    return counts


def sae_health() -> None:
    # Per-epoch R^2/L0/Dead lines from the train .out logs, keyed by array task
    # id in the filename (tf_k400_train_sae_<jobid>_<taskid>.out).
    health_re = re.compile(r"R²=([\d.]+)\s+MSE=([\d.]+)\s+L0=([\d.]+)\s+Dead=(\d+)")
    lines = ["# SAE health (per layer, per epoch — canonical copy in wandb)", ""]
    for out in sorted(ROOT.glob(CFG["train_out_glob"])):  # .out lands at repo root (SBATCH cwd)
        task = out.stem.split("_")[-1]
        epochs = health_re.findall(out.read_text())
        ckpt = torch.load(CFG["sae_dir"] / CFG["sae_pattern"].format(layer=int(task)),
                          map_location="cpu", weights_only=True)
        nb = ckpt["sae_state_dict"]["dictionary._weights"].shape[0]
        lines += [f"## Layer {task}  (sae_k={ckpt.get('sae_k')}, nb_concepts={nb}, "
                  f"best epoch score={ckpt.get('score'):.4f})", ""]
        lines += [f"- epoch {i+1}: R²={r}  MSE={m}  L0={l}  Dead={d}"
                  for i, (r, m, l, d) in enumerate(epochs)]
        lines.append("")
    (OUT / "sae_health.md").write_text("\n".join(lines) + "\n")


def spliced_table() -> None:
    lines = ["# Spliced accuracy (Job 3, same 3,000-clip sample as Job 0)", "",
             "| layer | baseline | spliced | drop |", "|---|---|---|---|"]
    for l in LAYERS:
        df = pd.read_csv(CFG["spliced_dir"] / CFG["spliced_pattern"].format(layer=l))
        row = df[df["template"] == "OVERALL_CLIP_WEIGHTED"].iloc[0]
        lines.append(f"| {l} | {row['baseline_accuracy']:.4f} | "
                     f"{row['spliced_accuracy']:.4f} | {row['delta']:+.4f} |")
    (OUT / "spliced_accuracy.md").write_text("\n".join(lines) + "\n")


def dfa_summary() -> None:
    lines = ["# DFA + four-bucket summary", ""]
    for l in LAYERS:
        n = len(pd.read_parquet(CFG["dfa_dir"] / CFG["dfa_pattern"].format(layer=l)))
        lines.append(f"- layer {l}: {n} R-correct clips in the per-clip raw parquet")
        bucket = CFG["bucket_dir"] / CFG["bucket_pattern"].format(layer=l)
        if bucket.exists():
            b = pd.read_csv(bucket)
            lines.append(f"  four-bucket: {len(b)} clips, {b['class_id'].nunique()} classes "
                         f"(>=40% R-acc eligible pool) -> {bucket.name}")
    (OUT / "dfa_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    verify_counts()
    sae_health()
    spliced_table()
    dfa_summary()
    print(f"Reports written under {OUT}")


if __name__ == "__main__":
    main()
