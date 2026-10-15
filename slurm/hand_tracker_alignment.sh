#!/bin/bash
#SBATCH --job-name=tot_hand_align
#SBATCH --output=hand_align_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --array=0-8

# Hand-tracker search: every feature of VM / TF / UMT at L5 / L7 / L9 scored against UMT L9 #2782
# on the 670 move up/down clips (src/stage3_analysis/hand_tracker_alignment.py).
# Task i: MODEL = (umt videomae timesformer)[i / 3], LAYER = (5 7 9)[i % 3].
# L5/L7: first extract R-only sparse codes (umt_vision_pair_contrast.py, LAYER, CONDS=R) into
# <model>_vision_pair_contrast_L<layer>/; L9 reuses the existing 12-13/10 extractions.
# Needs umt_vision_pair_contrast/sparse/R (the #2782 template). Output:
# outputs/analysis/hand_tracker_alignment/<model>_L<layer>.csv
MODELS=(umt videomae timesformer); LAYERS=(5 7 9)
export MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}
export LAYER=${LAYERS[$((SLURM_ARRAY_TASK_ID % 3))]}
export CONDS=R PAIR=45,43 MAX_CLIPS=0
echo "MODEL=$MODEL LAYER=$LAYER"

source $HOME/.tokens
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd $HOME/temporal-or-textural &&
    if [ \"$LAYER\" != 9 ]; then python src/stage3_analysis/umt_vision_pair_contrast.py; fi &&
    python src/stage3_analysis/hand_tracker_alignment.py
"
