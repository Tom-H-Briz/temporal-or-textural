#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_ssv2_x16_7ep
#SBATCH --output=train_sae_vm_ssv2_x16k128_7ep_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --array=5,7,9

# Runs: vm_ssv2_l5_x16k128_7ep, vm_ssv2_l7_x16k128_7ep, vm_ssv2_l9_x16k128_7ep.
# Extends the original L5-only capacity test to all three layers, and — like
# train_sae_vm_ssv2_l5_l7_l9.sh — is a full redo under the transformers==5.5.0 pin.
# See that script for the shared prereq/decision-rule/eval-methodology notes.
#
# 3h per-layer ceiling — same basis as the original L5-only script: x8k64's ~84min
# estimate padded for the larger dictionary's extra SAE-side compute, not a fresh
# measurement for x16k128 specifically.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=videomae
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=128
export SAE_EXPANSION=16
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_vmae_ssv2_k128_x16_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    export RESUME_FROM="$_CKPT"
    echo "Found existing checkpoint, resuming: $_CKPT"
fi
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/vmae_ssv2_layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/train_sae.py
    "
