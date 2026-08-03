#!/bin/bash
#SBATCH --job-name=tot_pos_lock_vm_kinetics
#SBATCH --output=position_lock_vm_kinetics_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=5,7,9

# Merged DFA + raw-activation extraction (31/07 consolidation) — replaces
# run_dfa_per_tubelet_mass_vm_kinetics.sh and run_z_position_lock_vm_kinetics.sh.
# K400 population = k400_manifest_SL_subset.json (Step 0a, 03/08 CC brief) —
# REQUIRES build_k400_sl_manifest.sh to have been run at least once already;
# this script no longer re-derives the population dynamically (that's the
# whole point of the manifest — see build_k400_sl_manifest.sh's comment).
# KINETICS_LABELS_CSV is not read by position_lock_extraction.py's K400 path
# any more (only build_k400_sl_manifest.sh's job needs it) — left unexported
# here to avoid implying it still does anything.
# Time budget doubled vs the SSv2 launcher — K400 clips run ~2x longer than SSv2
# clips, and decode cost scales with clip length even though the model still only
# samples a fixed 16 frames per clip.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/position_lock_extraction.py --model videomae --dataset kinetics400 --layer $SLURM_ARRAY_TASK_ID
    "
