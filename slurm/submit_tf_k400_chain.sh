#!/bin/bash
# One-shot chained submission for the TF-K400 run (brief, 02/10).
# Every job exits non-zero on gate failure; --dependency=afterok means a
# failure anywhere stops everything downstream. Run AFTER Gate 1's load report
# (tf_k400_gate1_load_report.sh) has been checked and appended to
# outputs/tf_k400/gate1_source.md.
#
#   J0 baseline gate ──> J1 dim_mean (5,7,9) ──> J2 SAE (5,7,9) ──┬─> J4b DFA (5,7,9) ──> J5 four-bucket + reports
#                     └────────────────────────────> J4a perturb ─┘
#
# Usage: bash slurm/submit_tf_k400_chain.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # submit from repo root so SBATCH --output lands there

J0=$(sbatch slurm/tf_k400_baseline_gate.sh        | awk '{print $NF}')
J1=$(sbatch --dependency=afterok:"$J0" slurm/tf_k400_dim_mean.sh          | awk '{print $NF}')
J2=$(sbatch --dependency=afterok:"$J1" slurm/tf_k400_train_sae.sh         | awk '{print $NF}')
J3=$(sbatch --dependency=afterok:"$J2" slurm/tf_k400_spliced_accuracy.sh  | awk '{print $NF}')
J4A=$(sbatch --dependency=afterok:"$J0" slurm/tf_k400_perturb_accuracy.sh | awk '{print $NF}')
# J4b needs BOTH the SAEs (J2) and, for its reliability context, nothing from J4a —
# but J5 needs J4a (class-pool CSV) and J4b (parquets), so J5 waits on both.
J4B=$(sbatch --dependency=afterok:"$J2" slurm/tf_k400_dfa_extraction.sh   | awk '{print $NF}')
J5=$(sbatch --dependency=afterok:"$J4A":"$J4B" slurm/tf_k400_four_bucket.sh | awk '{print $NF}')

cat <<EOF
Submitted TF-K400 chain:
  J0  baseline gate        $J0
  J1  dim_mean 5/7/9       $J1   (after $J0)
  J2  SAE train 5/7/9      $J2   (after $J1)
  J3  spliced 5/7/9        $J3   (after $J2)
  J4a perturb R/C/A        $J4A  (after $J0)
  J4b DFA stack 5/7/9      $J4B  (after $J2)
  J5  four-bucket+reports  $J5   (after $J4A and $J4B)
EOF
