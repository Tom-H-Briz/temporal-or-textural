#!/bin/bash
#SBATCH --job-name=tot_scaffold_abl_l5
#SBATCH --output=scaffold_ablation_l5_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

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
        pip install --quiet av einops pandas pyarrow transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        echo '=== run_ablation (L5 clean8) ===' &&
        python src/stage3_analysis/run_ablation.py --layer 5 --job-label 64 --sae-k 64 --run-tag l5_clean8_150726 &&
        echo '=== ablation_l5_vs_l7_comparison ===' &&
        python src/stage3_analysis/ablation_l5_vs_l7_comparison.py &&
        echo '=== done ==='
    "
