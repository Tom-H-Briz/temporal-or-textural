#!/bin/bash
#SBATCH --job-name=tot_perturb_acc_vivit_kinetics
#SBATCH --output=run_perturb_accuracy_vivit_kinetics_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# Conditions R/A/C1 for ViViT-B/16x2 (32 frames/clip, 2x the VM tokens per forward).
# Walltime: ViViT-kinetics R measured 49:59/condition on the stride-4 run — ~2.5h
# for 3 conditions + ~10min container/checkpoint load = ~3h estimate; 4h for
# safety headroom. R lands first regardless of any timeout.
# This re-run fixes the stride: frame_sample_rate 4 -> 2 (paper §4.1 protocol),
# replacing the stride-4 numbers (R top-1 0.5687, recorded in the stride-4 .out).
# R doubles as the label-map validation gate: ~0.25% top-1 means the canonical
# alphabetical fallback is wrong, ~60%+ means it's right (kill early if you see it).
#
# VIDEO_DIR/KINETICS_LABELS_CSV must be set explicitly — data/ is gitignored and
# never synced; DATASET_REGISTRY's repo-relative default doesn't exist on Isambard
# (5848353 failed in 31s on exactly this — see run_perturb_accuracy_vm_kinetics.sh).
# R runs first and its CSV hits disk before A/C1 start, so a timeout partway
# through A/C1 doesn't lose R.

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
        python notebooks/perturb_accuracy_vivit_kinetics.py
    "
