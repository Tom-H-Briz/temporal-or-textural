#!/bin/bash
#SBATCH --job-name=tot_sae_sweep
#SBATCH --output=sweep_dead_features_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-7

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

# Job configs indexed by SLURM_ARRAY_TASK_ID
#              A     B     C     D     E            F     G     H
K_ARR=(        64    64    64    64    64           96    128   64  )
ALPHA_ARR=(    0.03  0.1   0.3   1.0   0.03         0.03  0.03  0.03)
LOSS_ARR=(     aux   aux   aux   aux   reanimation  aux   aux   aux )
LABEL_ARR=(    A     B     C     D     E            F     G     H   )

export SAE_K="${K_ARR[$SLURM_ARRAY_TASK_ID]}"
export SAE_ALPHA="${ALPHA_ARR[$SLURM_ARRAY_TASK_ID]}"
export SAE_LOSS_FN="${LOSS_ARR[$SLURM_ARRAY_TASK_ID]}"
export SAE_JOB_LABEL="${LABEL_ARR[$SLURM_ARRAY_TASK_ID]}"
export SAE_EPOCHS=1

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
