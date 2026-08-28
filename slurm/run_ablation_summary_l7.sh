#!/bin/bash
#SBATCH --job-name=tot_ablation_summary_l7
#SBATCH --output=ablation_summary_l7_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

# CPU-only aggregation of run_ablation_l7.sh's output (mirrors
# make_probe_config.sh's resource shape — no --gpus, no --nv). Includes
# VM_TOP12 (27/08/26 CC brief) alongside the existing all4 target since both
# now live in ablation_targets.py's ("ssv2", 7) entry and run_ablation.py
# ablates everything in that dict in one pass. flip_rate IS the accuracy
# report (1 - correct_ablated.mean(), per condition/target/stratum).

source $HOME/.tokens

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet pandas pyarrow numpy &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/ablation_summary.py \
            outputs/analysis/scaffold_ablation/ablation_results_long_ssv2_l7_job7ep_k64.parquet
    "
