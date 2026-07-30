#!/bin/bash
#SBATCH --job-name=tot_check_k400_baseline
#SBATCH --output=check_k400_baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Quick sanity check: baseline-only accuracy on 3,000 random K400 clips under the
# corrected sample_frames_kinetics sampler, before resubmitting the full dim_mean +
# SAE retrain cycle. Compute itself is short (~7-8min: 3000/8=375 batches at the
# ~1.14s/batch rate observed in the earlier full spliced-accuracy run); the 30min
# ceiling mostly pads for model download + apptainer/pip startup, not the inference.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops transformers huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/check_kinetics_baseline.py --n-clips 3000
    "
