#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_k400_x16
#SBATCH --output=train_sae_vm_k400_x16k128_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=5,7,9

# Runs: vm_k400_l5_x16k128_7ep, vm_k400_l7_x16k128_7ep, vm_k400_l9_x16k128_7ep — not
# in the original design brief (20/07/26), added afterward to mirror the SSv2 L5
# capacity test (x8k64 vs x16k128) across all three K400 layers, not just L5.
#
# Prereqs: same as train_sae_vm_kinetics_l5_l7_l9.sh — dim_mean/VIDEO_DIR/labels CSV
# are all shared, since dim_mean depends on (model, dataset, layer) only, not
# expansion/k. No separate dim_mean sweep needed; the x8k64 sweep's already covers this.
#
# 10h ceiling, generously padded — this run's operator is unreachable for a week, so
# a single self-resuming submission has to have a strong chance of finishing in one
# shot rather than dying and sitting untouched. Basis: x8k64's real measured numbers
# (job 5730629/5733912: ~45min/epoch, spliced accuracy ~19min for baseline+spliced on
# the 3,976-clip held-out set) give ~5h34m for a fresh 7-epoch x8k64 run; padded ~30%
# for x16k128's larger dictionary (~7h15m), then padded again for the unattended week
# rather than trimmed tight. Not a fresh measurement for x16k128 specifically — tighten
# once a real x16k128 K400 run completes.
#
# Eval: real clips only (condition R) — no perturbed conditions, matching the x8k64
# K400 runs. 20% held-out split of the K400 val set, same clips serve SAE validation
# and spliced accuracy (train_sae.py persists the split).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export MODEL_NAME=videomae
export DATASET_NAME=kinetics400
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=128
export SAE_EXPANSION=16
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_VAL_FRACTION=0.2
# Self-resuming: if this exact script gets resubmitted after a time-limit kill, pick
# up from the rolling-latest checkpoint for this layer instead of restarting epoch 1.
# Safe on a fresh run too — file won't exist yet, so RESUME_FROM just stays unset.
_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_vmae_kinetics400_k128_x16_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    export RESUME_FROM="$_CKPT"
    echo "Found existing checkpoint, resuming: $_CKPT"
fi
export SAE_JOB_LABEL=7ep
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/vmae_kinetics400_layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"

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
