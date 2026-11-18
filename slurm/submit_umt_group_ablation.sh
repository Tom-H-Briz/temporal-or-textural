#!/bin/bash
set -euo pipefail
# Group + single ablation, then its analysis as a dependent job (18/11/26). Run on the login node from the repo root:
#   bash slurm/submit_umt_group_ablation.sh                      (defaults: top20_4250 on up/down 45/43, L5, spliced)
#   ARM_GROUPS=<key> CLASSES=<a,b> OUT_DIR=<dir> bash slurm/submit_umt_group_ablation.sh
# Then copy down only the summary: outputs/analysis/<run>/analysis/group_ablation.csv
export ARM_GROUPS=${ARM_GROUPS:-top20_4250} CLASSES=${CLASSES:-45,43} LAYER=${LAYER:-5}   # top 20 pos + 20 neg (18/11)
export OUT_DIR=${OUT_DIR:-outputs/analysis/umt_top20_4250_L5_ud_splice}
grep -q ARM_GROUPS src/stage3_analysis/umt_clock_downstream_ablation.py || { echo "driver has no ARM_GROUPS: git pull"; exit 1; }
if [ -f "$OUT_DIR/arms.csv" ] && grep -q ",ramp," "$OUT_DIR/arms.csv"; then
    echo "$OUT_DIR holds a run with the default arms (ramp, ...): remove it first"; exit 1
fi
J=$(sbatch --parsable --export=ALL slurm/umt_clock_downstream_ablation.sh)
A=$(sbatch --parsable --dependency=afterok:$J --export=ALL,RUN_DIR=$OUT_DIR slurm/umt_group_ablation_analyse.sh)
echo "ablation $J -> analysis $A (afterok) | ARM_GROUPS=$ARM_GROUPS CLASSES=$CLASSES OUT_DIR=$OUT_DIR"
