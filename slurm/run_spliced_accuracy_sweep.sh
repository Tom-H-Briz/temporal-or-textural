#!/bin/bash
#SBATCH --job-name=tot_spliced_sweep
#SBATCH --output=spliced_accuracy_sweep_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# Comprehensive spliced-accuracy sweep, all 14 current (post-30/07-fix) VideoMAE
# SAEs — SSv2 L3/L5/L7/L9 x k64/k128 (minus the L3/k128 cell, no checkpoint exists)
# + K400 same shape. One canonical held-out set per dataset (SSv2: generated this
# run via train_sae.py's own build_split, byte-identical to what real training
# used; K400: reuses the already-persisted 3,976-clip file). TimeSformer excluded
# — see spliced_accuracy_sweep.py's docstring.
#
# Both VIDEO_DIR and SSV2_VIDEO_DIR/K400_VIDEO_DIR are needed: VIDEO_DIR is read
# once at import time by train_sae.py's module-level CFG (used only for the SSv2
# held-out split generator, so it must point at the SSv2 dir); SSV2_VIDEO_DIR/
# K400_VIDEO_DIR are read fresh per config by spliced_accuracy_sweep.py itself,
# since run_spliced_accuracy's single VIDEO_DIR lookup can't serve both datasets
# within one process (see spliced_accuracy_sweep.py's CFG comment).

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export SSV2_VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export K400_VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/spliced_accuracy_sweep.py
    "
