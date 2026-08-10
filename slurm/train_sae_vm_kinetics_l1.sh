#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_k400_l1
#SBATCH --output=train_sae_vm_k400_l1_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=1

# L1 extension of train_sae_vm_kinetics_l5_l7_l9.sh — same 6h ceiling reasoning
# (K400 clips are full ~10s YouTube videos, decode-per-clip dominates over layer
# choice, so no time-budget change for L1 vs L5/L7/L9).
#
# Prereq: compute_dim_mean_vm_kinetics_l1.sh completed — writes
# vmae_kinetics400_layer1_dim_mean.pt, using the same VIDEO_DIR override as below.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export MODEL_NAME=videomae
export DATASET_NAME=kinetics400
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_VAL_FRACTION=0.2
# Self-resuming: same pattern as the sweep script.
_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_vmae_kinetics400_k64_x8_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    export RESUME_FROM="$_CKPT"
    echo "Found existing checkpoint, resuming: $_CKPT"
fi
export SAE_JOB_LABEL=7ep
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/vmae_kinetics400_layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"

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
