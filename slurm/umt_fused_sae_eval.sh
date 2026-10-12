#!/bin/bash
#SBATCH --job-name=tot_umt_fused_sae_eval
#SBATCH --output=umt_fused_sae_eval_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=9,10

# Eval-only re-run for the fused-text SAEs (L9, L10) trained by umt_fused_sae_train.sh (job 7188447).
# The train script can't be resubmitted for this: RESUME_FROM on a complete checkpoint trips the
# "resume checkpoint already complete" assert and the && chain stops before eval.
# Runs MODE=eval only: VTM splice gate + null-video verb-signature gate on the _best checkpoint.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MANIFEST_PATH="$HOME/temporal-or-textural/outputs/manifests/manifest_ssv2_all174_cap40.json"
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
export SPLICE_CAP=805   # full val split (~107 pair-class clips); 300 gave only ~3 per pair

REPO="$HOME/temporal-or-textural"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd $REPO &&
        MODE=eval python src/stage2_sae/train_sae_umt_fused.py
    "
