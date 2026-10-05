#!/bin/bash
#SBATCH --job-name=tot_umt_ssv2_dfa
#SBATCH --output=umt_ssv2_dfa_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=5,7,9

# Per-layer DFA stack over the SSv2 SL manifest (5,807 clips), mirroring TF-K400 Job 4b:
#   1. dfa_mass_delta_vm.py --model umt -> per-clip R/C/A parquet (outputs/analysis/dfa_mass_delta_umt/)
#   2. position_lock_extraction.py      -> per-frame DFA + raw-z stats (outputs/analysis/position_lock/)
# 10h: VM-SSv2 took 2h + 4h for these two; UMT has 1.5x the tokens.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
L=$SLURM_ARRAY_TASK_ID

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py \
            --model umt --dataset ssv2 --layer $L --job-label 7ep --sae-k 64 &&
        python src/stage3_analysis/dfa_extraction/position_lock_extraction.py \
            --model umt --dataset ssv2 --layer $L
    "
