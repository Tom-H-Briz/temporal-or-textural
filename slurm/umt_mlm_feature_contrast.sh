#!/bin/bash
#SBATCH --job-name=tot_umt_mlm_contrast
#SBATCH --output=umt_mlm_contrast_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# UMT MLM feature contrast (src/stage3_analysis/umt_mlm_feature_contrast.py), brief
# 191026: L9 SAE feature contributions to the refit MLM head's masked pair-word T_ab,
# R / C / Rev, Phase 3's six pairs' clips (both splits); then Phase 3 selection (R,
# selection clips) and comparison with Phase 3's T_vtm selections. Token-level sparse
# output ~6.5 GB -> outputs/analysis/umt_mlm_feature_contrast/ (home; smoke reports size).
# MODE=extract|select|compare|check|all (a failed stage can be resubmitted alone).
# Smoke (10 per class per split = 40 clips per pair):
#   sbatch --export=ALL,MAX_CLIPS=20 slurm/umt_mlm_feature_contrast.sh
# Full:                 sbatch slurm/umt_mlm_feature_contrast.sh
# Pair subset (semicolon-separated, no commas -- sbatch splits on commas):
#   sbatch --export=ALL,PAIRS="46_5;45_43" slurm/umt_mlm_feature_contrast.sh

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MAX_CLIPS=${MAX_CLIPS:-0}
export PAIRS=${PAIRS:-}
export MODE=${MODE:-all}
echo "MODE=$MODE MAX_CLIPS=$MAX_CLIPS PAIRS=$PAIRS"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python src/stage3_analysis/umt_mlm_feature_contrast.py \$MODE
    "
