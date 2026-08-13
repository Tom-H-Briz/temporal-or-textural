#!/bin/bash
#SBATCH --job-name=tot_dfa_mass_delta_vm_k128_l5_l7_l9
#SBATCH --output=dfa_mass_delta_vm_k128_l5_l7_l9_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=5,7,9

# Fresh job7ep/k128 dfa_mass_delta_vm_c1 run for L5/L7/L9 — the only k128 file
# on disk (dfa_mass_delta_vm_c1_l7_job128_16x_k128.parquet) predates the 30/07
# bias fix, so scaffold_selection_consolidated.py's ceiling_check/mass-pct
# figures for the k128 configs would otherwise pair fresh post-fix
# position-locked members against a stale pre-fix mass floor. Companion to
# run_position_lock_vm_ssv2_l{5,7,9}_k128.sh. Same time budget as the k64
# L5/L7 run — dict-size doubling only adds SAE encode/decode cost.

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
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_mass_delta_vm.py --layer \$SLURM_ARRAY_TASK_ID --job-label 7ep --sae-k 128
    "
