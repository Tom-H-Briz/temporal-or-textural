#!/bin/bash
#SBATCH --job-name=tot_dfa_tubelet_vm_scan
#SBATCH --output=dfa_per_tubelet_mass_vm_scan_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# Three VM SAE configs to confirm scaffold features exist under: L5/L9 (fresh, k=64,
# 8x) plus the pre-existing L7 k=128/16x checkpoint. Not a uniform layer sweep, so
# indexed via parallel bash arrays rather than --array=<layers>.
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
        python src/stage3_analysis/dfa_per_tubelet_mass.py --model videomae --layer $LAYER --job-label $JOB --sae-k $K
    "
