#!/bin/bash
#SBATCH --job-name=tot_dim_mean_vm_kinetics
#SBATCH --output=compute_dim_mean_vm_kinetics_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --array=5,7,9

# Prereq: Kinetics-400 val clips staged on scratch (confirmed via
# inspect_kinetics_data.py: flat directory, 19,881 clips, not under data/kinetics400/val/
# — DATASET_REGISTRY's default doesn't match, hence the VIDEO_DIR override below).
# labels_path is None (no SSv2-style JSON) by design; this script never needs labels.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export MODEL_NAME=videomae
export DATASET_NAME=kinetics400
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export SAE_LAYER=$SLURM_ARRAY_TASK_ID

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/profile_activations.py
    "
