#!/bin/bash
#SBATCH --job-name=tot_umt_top20_extract
#SBATCH --output=umt_top20_extract_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Slice target latents out of the 7 top-20 group-ablation runs (19/11/26) -> <run>/extract/prof_L{c}_{cond}.npy.
# sbatch slurm/umt_top20_extract_targets.sh
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    cd \$HOME/temporal-or-textural &&
    python -u src/stage3_analysis/umt_top20_extract_targets.py
"
