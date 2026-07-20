#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_ssv2_l5_x16_7ep
#SBATCH --output=train_sae_vm_ssv2_l5_x16k128_7ep_%A.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=22:00:00

# Run: vm_ssv2_l5_x16k128_7ep (design brief: SAE Training Sweep — Kinetics L5/7/9 +
# SSv2 L5 Capacity Test, 20/07/26). Follow-up to the 14/07/26 job64 capacity-vs-
# convergence question — this run isolates "more capacity" (double expansion/k vs
# job64) at the full 7-epoch protocol. Sibling to job64, not a replacement — see
# train_sae_vm_ssv2_l5_x8k64_7ep.sh for the "more training" half of the same test
# and the decision rule for job64 supersession.
#
# 22h estimate, not measured — no prior x16k128-at-L5 timing exists; padded above
# the x8k64-7ep budget for the larger dictionary's extra encode/decode cost.
#
# Prereq: outputs/sae/vmae_ssv2_layer5_dim_mean.pt already exists (reused from job64
# — dim_mean depends on model/dataset/layer only, not expansion/k).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=videomae
export SAE_LAYER=5
export SAE_K=128
export SAE_EXPANSION=16
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/vmae_ssv2_layer5_dim_mean.pt"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/train_sae.py
    "
