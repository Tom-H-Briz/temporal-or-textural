#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_ssv2_7ep
#SBATCH --output=train_sae_vm_ssv2_x8k64_7ep_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:30:00
#SBATCH --array=5,7,9

# Runs: vm_ssv2_l5_x8k64_7ep, vm_ssv2_l7_x8k64_7ep, vm_ssv2_l9_x8k64_7ep. Extends
# the original L5-only capacity test (design brief 20/07/26) to all three layers,
# and — more importantly — is a full redo under the transformers==5.5.0 pin: every
# prior VideoMAE result (job64 included) was trained on activations extracted with
# corrupted attention query/value biases (see check_model_load_report.py findings,
# ~29/07/26). Sibling to job64, not a replacement — job64 stays canonical pending
# supersession on spliced accuracy specifically, decision rule per the design brief.
#
# Prereq: compute_dim_mean_vm_sweep.sh (array=5,7,9) completed under the pin —
# writes vmae_ssv2_layer{5,7,9}_dim_mean.pt. This overwrites the old (pre-fix)
# dim_mean files, which is intended — dim_mean is deterministic given the same
# clips/seed/model, so recomputing under correct weights is the whole point.
#
# 2.5h per-layer ceiling — same basis as the original L5-only script (job64 sacct:
# 44min/5ep, scaled to 7ep + spliced accuracy, ~1.8x slack). Array --time is
# per-task, not summed, so this is unchanged by going from 1 layer to 3.
#
# Eval: reuses the existing full SSv2 val set for spliced accuracy — same
# methodology job64's original figure was measured on, for direct comparability.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

export MODEL_NAME=videomae
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
# Self-resuming, same pattern as the K400 scripts: if this exact script gets
# resubmitted after a time-limit kill, pick up from the rolling-latest checkpoint
# for this layer instead of restarting epoch 1. Safe on a fresh run too.
_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_vmae_ssv2_k64_x8_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    export RESUME_FROM="$_CKPT"
    echo "Found existing checkpoint, resuming: $_CKPT"
fi
export DIM_MEAN_PATH="$HOME/temporal-or-textural/outputs/sae/vmae_ssv2_layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/train_sae.py
    "
