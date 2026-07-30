#!/bin/bash
#SBATCH --job-name=tot_check_ssv2_baseline
#SBATCH --output=check_ssv2_baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Same check as check_baseline_accuracy_kinetics.sh, for SSv2 — quantify how much
# the transformers attention-bias bug (see that script's comment for the full
# writeup) affected SSv2 specifically, before deciding whether job64/the SSv2
# capacity-test runs need re-deriving. transformers pinned to 5.5.0, same reasoning.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/check_baseline_accuracy.py --dataset-name ssv2 --n-clips 3000
    "
