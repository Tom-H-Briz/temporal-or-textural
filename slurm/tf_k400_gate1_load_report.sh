#!/bin/bash
#SBATCH --job-name=tot_tf_k400_load_report
#SBATCH --output=tf_k400_gate1_load_report_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00

# Gate 1 item 5: load the TF-K400 checkpoint under the pinned transformers and
# let from_pretrained's own key report print. Run BEFORE the chain; any
# UNEXPECTED key = stop (append the table to outputs/tf_k400/gate1_source.md).
# transformers pinned 5.5.0 — the VideoMAE bias-name drift (5.8+) is VM-only,
# TimeSformer's fused-qkv is a different path, but the pin is kept project-wide.

source $HOME/.tokens   # exports HF_TOKEN

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/check_model_load_report.py --model-name timesformer --dataset-name kinetics400
    "
