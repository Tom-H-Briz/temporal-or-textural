#!/bin/bash
#SBATCH --job-name=dfa_mass_delta
#SBATCH --output=dfa_mass_delta_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

cd $HOME/temporal-or-textural

uv venv
source .venv/bin/activate
uv sync

export HF_TOKEN="your_token_here"
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"

uv run python src/stage3_analysis/dfa_mass_delta.py
