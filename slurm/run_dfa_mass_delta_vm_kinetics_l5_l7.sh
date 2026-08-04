#!/bin/bash
#SBATCH --job-name=tot_dfa_mass_delta_vm_kinetics_l5_l7
#SBATCH --output=dfa_mass_delta_vm_kinetics_l5_l7_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=5,7

# K400 equivalent of run_dfa_mass_delta_vm_l5_l7.sh (03/08 CC brief, Step 1) —
# unblocks run_ablation.py's K400 path and scaffold_selection_consolidated.py's
# K400 ceiling comparator (currently NaN, no mass-delta parquet exists yet).
# L9 excluded, same precedent as the SSv2 script: position_lock_extraction.py's
# 03/08 re-run confirmed 0 scaffold members at L9 (was already 0 pre-rerun too)
# — nothing to ablate, so no mass-delta run needed there either.
# Time budget doubled vs the SSv2 L5/L7 script, same reasoning as
# run_position_lock_vm_kinetics.sh: K400 clips run ~2x longer than SSv2 clips.
# Requires k400_manifest_SL_subset.json already present on Isambard (scp'd
# 03/08 — see Step 0) and the K400 SAE checkpoints (already there, used by
# position_lock_extraction.py's just-completed K400 run).

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_mass_delta_vm.py --dataset kinetics400 --layer \$SLURM_ARRAY_TASK_ID --job-label 7ep --sae-k 64
    "
