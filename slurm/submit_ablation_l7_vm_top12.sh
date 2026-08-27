#!/bin/bash
# Submits run_ablation_l7.sh (heavy, GPU — now includes VM_TOP12 alongside
# the existing all4 target, added to ablation_targets.py's ("ssv2", 7) entry),
# then run_ablation_summary_l7.sh (light, CPU — reports flip_rate/accuracy)
# chained on it via --dependency=afterok, so the accuracy job only starts
# once ablation finishes successfully.
#
# Not itself an sbatch script — run directly on the login node:
#   bash slurm/submit_ablation_l7_vm_top12.sh

cd "$(dirname "$0")"

ABLATION_JOBID=$(sbatch --parsable run_ablation_l7.sh)
echo "Submitted ablation (L7, all4 + VM_TOP12): $ABLATION_JOBID"

SUMMARY_JOBID=$(sbatch --parsable --dependency=afterok:$ABLATION_JOBID run_ablation_summary_l7.sh)
echo "Submitted accuracy/summary job: $SUMMARY_JOBID (waits on afterok:$ABLATION_JOBID)"
