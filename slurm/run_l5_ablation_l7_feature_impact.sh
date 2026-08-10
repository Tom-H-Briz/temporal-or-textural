#!/bin/bash
#SBATCH --job-name=tot_l5_l7_impact
#SBATCH --output=l5_ablation_l7_feature_impact_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Two forward passes per clip (baseline + L5-ablated), 4,450 clips — same
# order of magnitude as run_ablation_l5.sh's population, GPU should clear
# this well inside 2h even with the extra L7 SAE encode per pass.

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
        python src/stage3_analysis/l5_ablation_l7_feature_impact.py
    "
