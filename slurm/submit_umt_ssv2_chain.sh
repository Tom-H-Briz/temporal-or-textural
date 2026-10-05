#!/bin/bash
# UMT-SSv2 chain. afterok everywhere: a failure stops everything downstream.
#
#   J1 dim_mean (5,7,9) ──> J2 SAE (5,7,9) ──> J3 DFA stack (5,7,9) ──┐
#   J0 perturb R/C/A (or an existing job id) ──────────────────────────┴─> J4 four-bucket
#
# Usage: bash slurm/submit_umt_ssv2_chain.sh [PERTURB_JOBID]   # pass the id if perturb is already queued
set -euo pipefail
cd "$(dirname "$0")/.."   # submit from repo root so SBATCH --output lands there

J0=${1:-$(sbatch slurm/umt_ssv2_perturb_accuracy.sh | awk '{print $NF}')}
J1=$(sbatch slurm/umt_ssv2_dim_mean.sh                                | awk '{print $NF}')
J2=$(sbatch --dependency=afterok:"$J1" slurm/umt_ssv2_train_sae.sh    | awk '{print $NF}')
J3=$(sbatch --dependency=afterok:"$J2" slurm/umt_ssv2_dfa.sh          | awk '{print $NF}')
J4=$(sbatch --dependency=afterok:"$J0":"$J3" slurm/umt_ssv2_four_bucket.sh | awk '{print $NF}')

cat <<EOF
Submitted UMT-SSv2 chain:
  J0  perturb R/C/A        $J0
  J1  dim_mean 5/7/9       $J1
  J2  SAE train 5/7/9      $J2   (after $J1)
  J3  DFA stack 5/7/9      $J3   (after $J2)
  J4  four-bucket          $J4   (after $J0 and $J3)
EOF
