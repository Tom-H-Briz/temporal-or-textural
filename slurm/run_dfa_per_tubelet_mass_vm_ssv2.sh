#!/bin/bash
#SBATCH --job-name=tot_dfa_tubelet_vm_ssv2
#SBATCH --output=dfa_per_tubelet_mass_vm_ssv2_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=5,7,9

# SSv2 arm, post-30/07 bias fix + checkpoint-resolution consolidation — all three
# layers at k=64/x8/job7ep (resolve_sae_checkpoint's defaults), matching the TF
# array pattern (run_dfa_per_tubelet_mass_tf.sh). This is the first successful
# end-to-end run for either dataset, not a regression check — see project memory
# project_dfa_pipeline_first_run_not_regression.md.

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
        python src/stage3_analysis/dfa_per_tubelet_mass.py --model videomae --layer $SLURM_ARRAY_TASK_ID
    "
