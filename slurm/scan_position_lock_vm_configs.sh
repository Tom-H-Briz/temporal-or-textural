#!/bin/bash
#SBATCH --job-name=tot_pos_lock_vm_scan
#SBATCH --output=position_lock_vm_scan_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# Merged DFA + raw-activation extraction (31/07 consolidation) — replaces
# scan_dfa_per_tubelet_mass_vm_configs.sh and scan_z_position_lock_vm_configs.sh.
# Same three VM SAE configs: L5/L9 (k=64, 8x) plus the pre-existing L7 k=128/16x
# checkpoint — indexed via parallel bash arrays, not a uniform layer sweep.
LAYERS=(5 9 7)
KS=(64 64 128)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}

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
        pip install --quiet av einops pandas pyarrow "transformers==5.5.0" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/position_lock_extraction.py --model videomae --layer $LAYER --job-label 7ep --sae-k $K
    "
