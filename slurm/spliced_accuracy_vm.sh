#!/bin/bash
#SBATCH --job-name=tot_spliced_vm
#SBATCH --output=spliced_accuracy_vm_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=5,9

# Prereq: train_sae_vm_l5_l9.sh must have completed for these layers first
# (writes outputs/sae/sae_layer{5,9}_job64.pt + layer{5,9}_dim_mean.pt).

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
        python notebooks/spliced_accuracy_vm.py --layer $SLURM_ARRAY_TASK_ID --dataset-name ssv2 \
            --sae-checkpoint outputs/sae/sae_layer${SLURM_ARRAY_TASK_ID}_job64.pt
    "
