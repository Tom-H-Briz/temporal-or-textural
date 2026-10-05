#!/bin/bash
#SBATCH --job-name=tot_tf_maxact
#SBATCH --output=tf_max_act_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# Max-activating clip renders for the TF heavy dozen on BOTH datasets (same
# criterion: top-12 by DFA mass, L7):
#   K400  — outputs/analysis/max_activating_tf/kinetics400_l7/feature{F}/
#   SSv2  — outputs/analysis/max_activating_tf/ssv2_l7/feature{F}/
# One forward pass per clip across the full SL manifests (~3k K400 + ~3.5k
# SSv2), then top-5 renders per feature with class names + spatial overlays.
# 4h ceiling: ~2h scan + 120 renders + model download headroom.

source $HOME/.tokens   # exports HF_TOKEN

K400_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
SSV2_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        VIDEO_DIR=$K400_DIR \
            python src/stage3_analysis/visualisation/max_activating_clips_tf.py --dataset kinetics400 &&
        VIDEO_DIR=$SSV2_DIR LABELS_PATH=\$HOME/labels/labels.json VALIDATION_PATH=\$HOME/labels/validation.json \
            python src/stage3_analysis/visualisation/max_activating_clips_tf.py --dataset ssv2
    "
