#!/bin/bash
# Submits the K400 dim_mean sweep, then the K400 SAE training sweep chained on it via
# SLURM's job-array dependency — one command instead of waiting and resubmitting by hand.
#
# Not itself an sbatch script — run directly on the login node:
#   bash slurm/submit_kinetics_sweep.sh
#
# aftercorr (not afterok) matches array task IDs 1:1 (dim_mean layer 5 -> train layer 5,
# etc.), so each layer's training starts as soon as that layer's dim_mean succeeds,
# rather than all three training jobs waiting for the slowest dim_mean layer.

cd "$(dirname "$0")"

DIM_MEAN_JOBID=$(sbatch --parsable compute_dim_mean_vm_kinetics_sweep.sh)
echo "Submitted dim_mean sweep: $DIM_MEAN_JOBID"

TRAIN_JOBID=$(sbatch --parsable --dependency=aftercorr:$DIM_MEAN_JOBID train_sae_vm_kinetics_l5_l7_l9.sh)
echo "Submitted training sweep: $TRAIN_JOBID (each layer waits on dim_mean job $DIM_MEAN_JOBID, same layer)"
