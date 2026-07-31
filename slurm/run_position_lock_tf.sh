#!/bin/bash
#SBATCH --job-name=tot_pos_lock_tf
#SBATCH --output=position_lock_tf_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=5,7,9

# Merged DFA + raw-activation extraction (31/07 consolidation) — replaces
# run_dfa_per_tubelet_mass_tf.sh and run_z_position_lock_tf.sh.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow "transformers==5.5.0" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/position_lock_extraction.py --model timesformer --layer $SLURM_ARRAY_TASK_ID
    "
