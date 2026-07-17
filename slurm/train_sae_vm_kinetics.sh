#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_kinetics
#SBATCH --output=train_sae_vm_kinetics_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=5,7,9

# Prereq: compute_dim_mean_vm_kinetics_sweep.sh must have completed for these layers
# first (writes outputs/sae/vmae_kinetics400_layer{5,7,9}_dim_mean.pt).
# No VIDEO_DIR/LABELS_PATH override — DATASET_REGISTRY["kinetics400"] resolves paths.
# No probe / DFA consumption this round — R2 and dead-feature fraction only.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export MODEL_NAME=videomae
export DATASET_NAME=kinetics400
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7                        # matches TF l3/l7/l9 7-epoch protocol — run fresh, no resume-from-shorter
export SAE_JOB_LABEL=$SLURM_ARRAY_TASK_ID  # label = layer number, readable in wandb

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/train_sae.py
    "
