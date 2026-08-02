#!/bin/bash
#SBATCH --job-name=tot_ablation_cross_l5_l7
#SBATCH --output=ablation_cross_l5_l7_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Both L5 and L7 SAEs spliced simultaneously, ablating the combined 11-feature
# set (L5's all7 + L7's all4) in one forward pass — R condition only. Standalone
# script (ablation_cross_l5_l7.py), not run_ablation.py — DFAEngine only splices
# one layer at a time. Source: SL manifest (~5807 clips), R-correctness
# recomputed fresh under the dual-spliced baseline.

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
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/ablation_cross_l5_l7.py
    "
