#!/bin/bash
#SBATCH --job-name=tot_umt_ssv2_train_sae
#SBATCH --output=umt_ssv2_train_sae_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --array=5,7,9

# UMT-SSv2 SAEs, config identical to VM-SSv2 (train_sae_vm_ssv2_l5_l7_l9.sh): k=64,
# x8, alpha=0.03, aux loss, 7 epochs, job label 7ep, legacy 20k/4k split (no
# SAE_VAL_FRACTION) -> epilogue spliced accuracy on full SSv2 val, as VM/TF-SSv2.
# 5h vs VM's 2.5h: 2352 tokens/clip vs 1568, plus the full-val epilogue.
# Self-resuming: resubmit THIS script after a time-limit kill (not the chain).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MODEL_NAME=umt
export DATASET_NAME=ssv2
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/umt_ssv2_layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"

_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_umt_ssv2_k64_x8_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    echo "RESUME_FROM=$_CKPT"
    export RESUME_FROM="$_CKPT"
fi

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python src/stage2_sae/train_sae.py
    "
