#!/bin/bash
#SBATCH --job-name=tot_dim_mean_vm_l3
#SBATCH --output=compute_dim_mean_vm_l3_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --array=3

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=videomae          # VM-only job — prereq for train_sae_vm_ssv2_l3.sh
export SAE_LAYER=$SLURM_ARRAY_TASK_ID

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/profile_activations.py
    "
