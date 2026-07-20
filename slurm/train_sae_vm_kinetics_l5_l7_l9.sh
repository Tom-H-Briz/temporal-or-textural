#!/bin/bash
#SBATCH --job-name=tot_train_sae_vm_k400
#SBATCH --output=train_sae_vm_k400_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=19:00:00
#SBATCH --array=5,7,9

# Runs: vm_k400_l5_x8k64_7ep, vm_k400_l7_x8k64_7ep, vm_k400_l9_x8k64_7ep
# (design brief: SAE Training Sweep — Kinetics L5/7/9 + SSv2 L5 Capacity Test, 20/07/26).
# First SAEs trained on the K400 backbone — tests whether L5/L7 scaffold features are
# MAE-pretraining artifacts or specifically demanded by SSv2's temporal task.
#
# Prereqs:
#   - compute_dim_mean_vm_kinetics_sweep.sh completed for these layers, using the
#     same VIDEO_DIR override as below (writes vmae_kinetics400_layer{5,7,9}_dim_mean.pt)
#   - VIDEO_DIR confirmed via inspect_kinetics_data.py: flat directory, 19,881 clips,
#     {youtube_id}_{start:06d}_{end:06d}.mp4 filenames — matches assumed convention
#   - val.csv confirmed present alongside the clips, header:
#     label,youtube_id,time_start,time_end,split,is_cc (matches assumed DeepMind format)
#
# 19h estimate, not measured — scaled from the SSv2 12h/5ep budget for 7 epochs plus
# the now-integrated spliced-accuracy pass on the held-out 20% split. Time budget is
# a safety ceiling only; Tom watches WandB and decides whether a run needs requeuing.
#
# Eval: real clips only (condition R) — no perturbed conditions. WebM/VP8-VP9->h264
# codec branch for Kinetics mp4s isn't built (future_work.md), so B/A/C aren't
# available regardless of eval-set design. 20% held-out split of the K400 val set,
# random (not class-stratified, matching SSv2 precedent) — same clips serve SAE
# validation and spliced accuracy, no separate pool (train_sae.py persists the split).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export MODEL_NAME=videomae
export DATASET_NAME=kinetics400
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_VAL_FRACTION=0.2
export SAE_JOB_LABEL=7ep
# DATASET_REGISTRY["kinetics400"]["video_dir"] default (data/kinetics400/val) does
# not match the real layout — override explicitly, same as the SSv2 scripts do.
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
# val.csv already lives alongside the clips (the downloader tool's own manifest) —
# use it directly rather than staging a separate copy under data/kinetics400/.
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
