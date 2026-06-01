#!/bin/bash
#SBATCH --job-name=tot_feat_counts
#SBATCH --output=extract_feature_counts_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

# Set these to match the run you want to inspect
export SAE_K=64
export SAE_JOB_LABEL=64

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/extract_feature_counts.py
    "
