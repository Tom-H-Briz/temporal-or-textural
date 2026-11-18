#!/bin/bash
#SBATCH --job-name=tot_umt_grp_analyse
#SBATCH --output=umt_group_analyse_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Analysis of a umt_clock_downstream_ablation.py ARM_GROUPS run (18/11/26): -> $RUN_DIR/analysis/group_ablation.csv.
# Submitted by slurm/submit_umt_group_ablation.sh as a dependent job; RUN_DIR = the ablation job's OUT_DIR.
[ -z "$RUN_DIR" ] && { echo "RUN_DIR not set"; exit 1; }
export RUN_DIR
echo "RUN_DIR=$RUN_DIR"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet pandas pyarrow &&
    cd \$HOME/temporal-or-textural &&
    python -u src/stage3_analysis/umt_group_ablation_analyse.py
"
