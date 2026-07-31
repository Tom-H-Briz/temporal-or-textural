#!/bin/bash
#SBATCH --job-name=tot_perturb_acc_vm_kinetics
#SBATCH --output=run_perturb_accuracy_vm_kinetics_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# Conditions R/A/C1. VIDEO_DIR/KINETICS_LABELS_CSV must be set explicitly — unlike
# train_sae_vm_kinetics.sh (which never calls load_kinetics_metadata), this script
# does, and DATASET_REGISTRY's repo-relative default (data/kinetics400/val) doesn't
# exist on Isambard at all (data/ is gitignored, never synced). 5848353 failed in
# 31s on exactly this — confirmed 31/07 the real files live under /scratch.
#
# Time budget: each condition measured at ~48min on the full K400 clip set (much
# slower per-clip than SSv2 — longer clips to decode) — 1hr/condition x 3 conditions,
# not 2hr flat. R runs first (see main()'s loop order) and its CSV is written to
# disk before A/C1 start, so even a timeout partway through A/C1 doesn't lose R —
# the condition Tom needs most right now — but budget for all three regardless.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/perturb_accuracy_vm_kinetics.py
    "
