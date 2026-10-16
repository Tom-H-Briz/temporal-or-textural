#!/bin/bash
#SBATCH --job-name=tot_umt_mlm_refit
#SBATCH --output=umt_mlm_refit_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# UMT MLM head refit (src/stage3_analysis/umt_mlm_refit.py), brief 151026:
# extract per-blank layer-11 h (train R; val R/null/Rev), refit the ~623k-param
# MLM head on the cache, evaluate on val. Same env as umt_vtm_gate.sh + wandb,
# timm/easydict (vision tower). Caches (~3 GB) go to scratch, small outputs to
# outputs/analysis/umt_mlm_refit[_smoke]/.
# Smoke:  sbatch --export=ALL,MAX_TRAIN_CLIPS=500,MAX_VAL_CLIPS=200 slurm/umt_mlm_refit.sh
# Full:   sbatch slurm/umt_mlm_refit.sh
# Re-run a stage only (caches persist on scratch): MODE=train or MODE=eval
#   sbatch --export=ALL,MODE=eval slurm/umt_mlm_refit.sh
# Assumes $HOME/labels/train.json exists (SSv2 train captions; locally
# data/ssv2/labels/train.json).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export TRAIN_PATH="$HOME/labels/train.json"
export CACHE_DIR="$SCRATCHDIR/umt_mlm_refit"
export MODE=${MODE:-all}
[ -f "$TRAIN_PATH" ] || { echo "missing $TRAIN_PATH"; exit 1; }
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow wandb \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python src/stage3_analysis/umt_mlm_refit.py \$MODE
    "
