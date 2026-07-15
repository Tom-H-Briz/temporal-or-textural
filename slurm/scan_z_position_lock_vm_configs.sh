#!/bin/bash
#SBATCH --job-name=tot_z_pos_lock_vm_scan
#SBATCH --output=z_position_lock_vm_scan_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --array=0-2

# Same three VM SAE configs as scan_dfa_per_tubelet_mass_vm_configs.sh — see that
# script's comment for why this is indexed rather than a plain layer array.
LAYERS=(5 9 7)
JOBS=(64 64 128_16x)
KS=(64 64 128)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}
JOB=${JOBS[$SLURM_ARRAY_TASK_ID]}
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
        pip install --quiet av einops pandas pyarrow transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/z_position_lock_extraction.py --model videomae --layer $LAYER --job-label $JOB --sae-k $K
    "
