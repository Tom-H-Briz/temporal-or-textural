#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_ssv2_l3_7ep
#SBATCH --output=train_sae_vm_ssv2_l3_x8k64_7ep_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --array=3

# L3 extension of train_sae_vm_ssv2_l5_l7_l9.sh — same job7ep/transformers==5.5.0
# setup, one more layer point on the VM-SSv2 side.
#
# Prereq: compute_dim_mean_vm_l3.sh completed — writes vmae_ssv2_layer3_dim_mean.pt.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=videomae
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
# Self-resuming, same pattern as the sweep scripts: if this exact script gets
# resubmitted after a time-limit kill, pick up from the rolling-latest checkpoint
# for this layer instead of restarting epoch 1. Safe on a fresh run too.
_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_vmae_ssv2_k64_x8_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
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
        python src/stage2_sae/train_sae.py
    "
