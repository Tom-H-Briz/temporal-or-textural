#!/bin/bash
#SBATCH --job-name=tot_umt_vis_pair
#SBATCH --output=umt_vis_pair_%A.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# UMT vision-side pair contrast, Stage 1 (src/stage3_analysis/umt_vision_pair_contrast.py):
# move up (45) vs move down (43), all val clips, conditions R / C / Rev, vision L9 SAE at the
# TRAINED threshold (DFAEngine fix 12/10/26). Token-level sparse z + contrast DFA, ~1.8 GB.
# Smoke: sbatch --export=ALL,MAX_CLIPS=8 slurm/umt_vision_pair_contrast.sh
# Other backbones (same script): sbatch --export=ALL,MODEL=videomae slurm/umt_vision_pair_contrast.sh
#   (MODEL=umt|videomae|timesformer; VM shuffle = C1). Full runs also aggregate + ramp-check.

source $HOME/.tokens
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MAX_CLIPS=${MAX_CLIPS:-0}
export MODEL=${MODEL:-umt}
# Comma-free pair passing: sbatch --export splits on commas, so PAIR=42,40 breaks and the
# extractor silently falls back to the default 45,43 up/down dir (two such jobs racing on
# the same sparse/ files corrupt npz). Submit as:
#   sbatch --export=ALL,PAIR_A=42,PAIR_B=40 slurm/umt_vision_pair_contrast.sh
# PAIR_A/PAIR_B (explicit, per-submission) beat a stale exported PAIR left in the shell.
[ -n "$PAIR_A" ] && export PAIR="${PAIR_A},${PAIR_B:-43}"
export PAIR="${PAIR:-45,43}"
echo "PAIR=$PAIR MODEL=$MODEL MAX_CLIPS=$MAX_CLIPS AGG_ONLY=${AGG_ONLY:-0}"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd $HOME/temporal-or-textural &&
        [ -n \"$AGG_ONLY\" ] || python src/stage3_analysis/umt_vision_pair_contrast.py &&
        if [ \"\$MAX_CLIPS\" = 0 ]; then
            python src/stage3_analysis/umt_vision_pair_aggregate.py &&
            python src/stage3_analysis/vision_pair_ramp_check.py
        fi
    "
