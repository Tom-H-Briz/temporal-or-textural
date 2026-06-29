#!/bin/bash
#SBATCH --job-name=tot_dfa_vm_full
#SBATCH --output=dfa_vm_full_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

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
        pip install --quiet av einops pandas pyarrow matplotlib transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        echo '=== dfa_extraction_vm ===' &&
        python src/stage3_analysis/dfa_extraction_vm.py &&
        echo '=== dfa_mass_delta_vm ===' &&
        python src/stage3_analysis/dfa_mass_delta_vm.py
    "
