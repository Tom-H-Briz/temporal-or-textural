#!/bin/bash
# Submits the SSv2 dim_mean sweep, then both SSv2 SAE training sweeps (x8k64 and
# x16k128) chained on it via SLURM's job-array dependency. Mirrors
# submit_kinetics_sweep.sh exactly — see that script for the aftercorr reasoning.
#
# Not itself an sbatch script — run directly on the login node:
#   bash slurm/submit_ssv2_sweep.sh

cd "$(dirname "$0")"

DIM_MEAN_JOBID=$(sbatch --parsable compute_dim_mean_vm_sweep.sh)
echo "Submitted dim_mean sweep: $DIM_MEAN_JOBID"

X8_JOBID=$(sbatch --parsable --dependency=aftercorr:$DIM_MEAN_JOBID train_sae_vm_ssv2_l5_l7_l9.sh)
echo "Submitted x8k64 training sweep: $X8_JOBID (each layer waits on dim_mean job $DIM_MEAN_JOBID, same layer)"

X16_JOBID=$(sbatch --parsable --dependency=aftercorr:$DIM_MEAN_JOBID train_sae_vm_ssv2_x16k128_l5_l7_l9.sh)
echo "Submitted x16k128 training sweep: $X16_JOBID (each layer waits on dim_mean job $DIM_MEAN_JOBID, same layer)"
