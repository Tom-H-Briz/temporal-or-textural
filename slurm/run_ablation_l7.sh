#!/bin/bash
#SBATCH --job-name=tot_ablation_l7
#SBATCH --output=ablation_l7_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Explicit --layer 7 — run_ablation.py's default IS layer 7, but pinning it
# here anyway for symmetry with run_ablation_l5.sh and to guard against the
# default silently changing later. TARGETS in ablation_targets.py currently
# holds L7's 4 features (3347/5165/6021/6032 — 3 members + the 3347 near-miss).
# Not chained into ablation_summary.py — run that as a separate step once this
# output lands, same as the L5 run.

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
        python src/stage3_analysis/run_ablation.py --layer 7 --job-label 7ep --sae-k 64
    "
