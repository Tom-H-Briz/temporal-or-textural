#!/bin/bash
#SBATCH --job-name=tot_ablation_l5
#SBATCH --output=ablation_l5_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Explicit --layer 5 — run_ablation.py defaults to layer 7, and TARGETS in
# ablation_targets.py currently holds L5's fresh feature IDs (single active
# set by design, see that file's docstring). Not chained into
# ablation_summary.py — that script is unreviewed, run it as a separate step
# once this output lands.

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
        python src/stage3_analysis/run_ablation.py --layer 5 --job-label 7ep --sae-k 64
    "
