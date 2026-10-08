#!/bin/bash
#SBATCH --job-name=tot_tf_k400_baseline
#SBATCH --output=tf_k400_baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Job 0 — baseline gate. 3,000 random K400 val clips, seed 42, through the real
# run_spliced_accuracy path (baseline_only). First run scored 0.7290 vs the
# original 0.730 (= published 78.0 − 5pp): a 3-clip shortfall, ~0.12σ of the
# 0.81pp binomial SE, with both deliberate protocol deltas (single centre-crop
# vs 3-crop; VM-matched 64-frame window vs native 8×32) pushing down. Accepted
# by Tom 02/10 — threshold of record lowered to 0.720, which any real bug class
# (labels/normalisation/sampler) still misses by tens of points. Sample + report
# from the 0.7290 run are the ones the chain consumes.
# Exits non-zero below threshold -> downstream jobs never start.
# Also persists outputs/tf_k400/eval_clips_3000.json (Job 3 reuses this exact
# sample) and logs sampler frame indices for 3 clips into baseline_report.md.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/check_baseline_accuracy.py --model-name timesformer \
            --dataset-name kinetics400 --n-clips 3000 --threshold 0.720
    "
