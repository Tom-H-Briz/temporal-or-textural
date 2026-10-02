#!/bin/bash
#SBATCH --job-name=tot_tf_k400_baseline
#SBATCH --output=tf_k400_baseline_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Job 0 — baseline gate. 3,000 random K400 val clips, seed 42, through the real
# run_spliced_accuracy path (baseline_only). Threshold 0.730 = published 78.0
# top-1 (paper Table 5) minus 5pp, allowing for single centre-crop vs 3-crop.
# Exits non-zero below threshold -> downstream jobs (dim_mean etc.) never start.
# Also persists outputs/tf_k400/eval_clips_3000.json (Job 3 reuses this exact
# sample) and logs sampler frame indices for 3 clips into baseline_report.md.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/check_baseline_accuracy.py --model-name timesformer \
            --dataset-name kinetics400 --n-clips 3000 --threshold 0.730
    "
