#!/bin/bash
#SBATCH --job-name=tot_perturb_acc_vm_ssv2
#SBATCH --output=run_perturb_accuracy_vm_ssv2_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# R/A/C1 only (perturb_accuracy_vm.py computes A/B/C/C1/C2, deliberately never R —
# not what's needed here, see perturb_accuracy_vm_ssv2.py's own docstring).

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
        pip install --quiet av einops pandas pyarrow "transformers==5.5.0" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/perturb_accuracy_vm_ssv2.py
    "
