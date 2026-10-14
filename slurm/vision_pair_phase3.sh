#!/bin/bash
#SBATCH --job-name=tot_vp3
#SBATCH --output=vp3_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:30:00

# Phase 3 classifier port (src/stage3_analysis/vision_pair_phase3.py): all 8 pairs in one job
# (model loaded once, ~3.6k clips), then `combine` writes <run>/summary.csv and *_all.csv.
# All 8:    sbatch --export=ALL,MODEL=videomae,LAYER=9 slurm/vision_pair_phase3.sh
#           sbatch --export=ALL,MODEL=timesformer,LAYER=9 slurm/vision_pair_phase3.sh
# One pair: sbatch --export=ALL,MODEL=videomae,PAIR_INDEX=4 slurm/vision_pair_phase3.sh
#   (index into CFG["pairs"]: 0 cover/uncover, 1 open/close, 2 fold/unfold, 3 camera up/down,
#    4 move up/down, 5 towards/away camera, 6 closer to/away from, 7 closer/apart each other)
# Smoke:    sbatch --export=ALL,MODEL=videomae,PAIR_INDEX=4,MAX_CLIPS=40 slurm/vision_pair_phase3.sh
# MODEL=videomae|timesformer|umt (umt = its VTC classifier head); LAYER = any layer with a k64 SAE.

source $HOME/.tokens
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MODEL=${MODEL:-videomae}
export LAYER=${LAYER:-9}
export STAGE=${STAGE:-all}
export MAX_CLIPS=${MAX_CLIPS:-0}
PAIRS_ARG=$([ -n "$PAIR_INDEX" ] && echo "--pair-index $PAIR_INDEX" || echo "--all-pairs")
echo "MODEL=$MODEL LAYER=$LAYER $PAIRS_ARG STAGE=$STAGE MAX_CLIPS=$MAX_CLIPS"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/vision_pair_phase3.py $PAIRS_ARG
    "
