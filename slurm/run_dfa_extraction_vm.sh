#!/bin/bash
#SBATCH --job-name=tot_dfa_extraction_vm
#SBATCH --output=dfa_extraction_vm_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_extraction_vm.py
    "
