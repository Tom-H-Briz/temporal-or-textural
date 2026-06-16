#!/bin/bash
#SBATCH --job-name=class_selection_tf
#SBATCH --output=class_selection_tf_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

source $HOME/.tokens

cd $HOME/temporal-or-textural

# Point data dir at scratch without touching the script's ROOT-relative paths
mkdir -p data/ssv2
ln -sfn /scratch/b5bg/tomheslin83.b5bg/videos data/ssv2/20bn-something-something-v2
ln -sfn $HOME/labels/labels.json data/ssv2/labels/labels.json 2>/dev/null || true
ln -sfn $HOME/labels/validation.json data/ssv2/labels/validation.json 2>/dev/null || true

uv venv
source .venv/bin/activate
uv sync

uv run python notebooks/class_selection.py
