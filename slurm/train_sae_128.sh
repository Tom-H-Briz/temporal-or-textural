#!/bin/bash
#SBATCH --job-name=tot_train_sae
#SBATCH --output=train_sae_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=timesformer
export SAE_K=128
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=5
export SAE_JOB_LABEL=128

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib "transformers==5.5.0" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage2_sae/train_sae.py
    "