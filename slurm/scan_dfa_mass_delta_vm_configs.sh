#!/bin/bash
#SBATCH --job-name=tot_dfa_mass_delta_vm_scan
#SBATCH --output=dfa_mass_delta_vm_scan_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=0-2

# Same indexing scheme as scan_dfa_per_tubelet_mass_vm_configs.sh. job_label is "7ep"
# for all post-30/07 checkpoints (epoch-count suffix, no longer k/expansion-encoding
# like the old "64"/"128_16x" strings) — only sae_k varies here. Note: the L7/job64
# baseline this comment used to reference (dfa_mass_delta_vm_c1.parquet, unsuffixed)
# predates the 30/07 bias fix and is superseded, not a still-valid prior result.
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
        pip install --quiet av einops pandas pyarrow matplotlib "transformers==5.5.0" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_mass_delta_vm.py --layer $LAYER --job-label 7ep --sae-k $K
    "
