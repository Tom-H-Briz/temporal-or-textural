#!/bin/bash
#SBATCH --job-name=tot_pos_lock_vm_kinetics
#SBATCH --output=position_lock_vm_kinetics_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=5,7,9

# Merged DFA + raw-activation extraction (31/07 consolidation) — replaces
# run_dfa_per_tubelet_mass_vm_kinetics.sh and run_z_position_lock_vm_kinetics.sh.
# K400 class scope = the full SL-matched set (up to 64 classes), not yet filtered
# to the per-class accuracy >= 40% eligible subset — same pending follow-up noted
# in the pre-merge scripts. Needs k400_sl_class_mapping.csv present on Isambard.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

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
