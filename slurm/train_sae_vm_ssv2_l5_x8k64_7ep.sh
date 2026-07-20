#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_ssv2_l5_7ep
#SBATCH --output=train_sae_vm_ssv2_l5_x8k64_7ep_%A.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=19:00:00

# Run: vm_ssv2_l5_x8k64_7ep (design brief: SAE Training Sweep — Kinetics L5/7/9 +
# SSv2 L5 Capacity Test, 20/07/26). Follow-up to the 14/07/26 job64 capacity-vs-
# convergence question (job64: L5, x8k64, 5 epochs, 9.2pp clip-weighted spliced-
# accuracy drop) — this run isolates "more training" at the full 7-epoch protocol.
# Sibling to job64, not a replacement: writes a new checkpoint identity
# (sae_vmae_ssv2_k64_x8_l5_job7ep*.pt); job64 stays canonical pending supersession
# on spliced accuracy specifically (see decision rule in the design brief).
#
# Prereq: outputs/sae/vmae_ssv2_layer5_dim_mean.pt already exists (reused from job64
# — dim_mean depends on model/dataset/layer only, not expansion/k).
#
# Eval: reuses the existing full SSv2 val set for spliced accuracy — same methodology
# job64's 9.2pp figure was measured on, for direct comparability.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=videomae
export SAE_LAYER=5
export SAE_K=64
export SAE_EXPANSION=8
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
