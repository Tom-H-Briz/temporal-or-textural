#!/bin/bash
#SBATCH --job-name=tot_train_sae
#SBATCH --output=train_sae_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

module load cuda/12.6

cd $HOME/temporal-or-textural

uv venv
source .venv/bin/activate
uv sync

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

uv run python notebooks/train_sae.py
