#!/bin/bash
#SBATCH --job-name=tot_probe_config
#SBATCH --output=make_probe_config_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av pandas pyarrow transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/make_probe_config.py
    "
