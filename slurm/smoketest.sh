#!/bin/bash
#SBATCH --job-name=tot_smoketest
#SBATCH --output=smoketest_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

cd $HOME/temporal-or-textural

uv venv
source .venv/bin/activate
uv sync

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/ssv2_val_set"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

uv run python notebooks/train_sae_smoketest.py
