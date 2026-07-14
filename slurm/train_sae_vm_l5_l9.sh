#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm
#SBATCH --output=train_sae_vm_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=5,9

# Prereq: compute_dim_mean_vm_sweep.sh must have completed for these layers first
# (writes outputs/sae/layer{5,9}_dim_mean.pt).

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
export SAE_EPOCHS=5                 # 5 epochs x 20k train_clips = 100k clip exposures, matches L7 recipe
export SAE_JOB_LABEL=64             # matches legacy L7 checkpoint's job64 naming
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/train_sae.py &&
        cp outputs/sae/sae_vmae_k64_x8_l${SLURM_ARRAY_TASK_ID}_job64_best.pt outputs/sae/sae_layer${SLURM_ARRAY_TASK_ID}_job64.pt
    "
