#!/bin/bash
#SBATCH --job-name=tot_sl35_list
#SBATCH --output=sl35_list_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:10:00

# SL-35 clip list (25/11/26) -> outputs/analysis/sl35_clips.csv; the SL characterisation jobs depend on it.
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet pandas pyarrow &&
    cd \$HOME/temporal-or-textural &&
    python -u src/stage3_analysis/sl35_clip_list.py
"
