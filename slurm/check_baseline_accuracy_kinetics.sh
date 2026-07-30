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
#
# transformers pinned to 5.5.0 — an unpinned install drifted past a breaking change
# (somewhere between 5.5.0 and 5.8.0) in VideoMAE's attention bias parameter names
# (q_bias/v_bias -> query.bias/key.bias/value.bias), silently discarding this
# checkpoint's trained query/value biases on load and replacing them with default
# init. 5.5.0 confirmed correct (verified against the raw source directly); 5.8.0+
# confirmed broken. TimeSformer's fused-qkv attention is a different code path,
# unaffected either way. torch>=2.4 required by 5.5.0 — should be satisfied by
# this container.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops "transformers==5.5.0" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/check_baseline_accuracy.py --dataset-name kinetics400 --n-clips 3000
    "
