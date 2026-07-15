#!/bin/bash
#SBATCH --job-name=tot_dfa_mass_delta_vm_scan
#SBATCH --output=dfa_mass_delta_vm_scan_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2

# L7/job64 baseline already exists (outputs/analysis/dfa_mass_delta_vm_c1/dfa_mass_delta_vm_c1.parquet)
# — only the three new configs need generating. Same indexing scheme as
# scan_dfa_per_tubelet_mass_vm_configs.sh.
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
        pip install --quiet av einops pandas pyarrow matplotlib transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_mass_delta_vm.py --layer $LAYER --job-label $JOB --sae-k $K
    "
