#!/bin/bash
#SBATCH --job-name=tot_pos_lock_vm_kinetics_l3
#SBATCH --output=position_lock_vm_kinetics_l3_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=3

# L3 extension of run_position_lock_vm_kinetics.sh — same merged DFA + raw-
# activation extraction, one more layer point. Independent of dfa_mass_delta_vm.py
# (does its own extraction from scratch) — submitted with --dependency=afterok on
# the L3 kinetics mass-delta job for scheduling order, not a hard requirement.
# Requires build_k400_sl_manifest.sh to have already run (already satisfied,
# same manifest used throughout this session's K400 work).

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
