#!/bin/bash
#SBATCH --job-name=tot_umt_p3
#SBATCH --output=umt_p3_%A.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# UMT vision->text bridge, Phase 3 (brief dfa_attention_bridge_brief_121026.md):
# Stage A: G1b extended gradient path checks (A: forward identity, B: fp64 FD)
# Stage B: AP extraction for T_vtm, T_match10, T_match9 across all 3,647 clips
# Stage C/D: G3 agreement & G4 causal effect / L9-L10 key-val route split on ~300 clips
# Stage E: G2 completeness and gate aggregations.
# Smoke: sbatch --export=ALL,MAX_CLIPS=20 slurm/umt_bridge_phase3.sh
# Full:  sbatch slurm/umt_bridge_phase3.sh

source $HOME/.tokens
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export STAGE=${STAGE:-all}
export MAX_CLIPS=${MAX_CLIPS:-0}
export G34_CLIPS=${G34_CLIPS:-300}

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd $HOME/temporal-or-textural &&
        STAGE=$STAGE MAX_CLIPS=$MAX_CLIPS G34_CLIPS=$G34_CLIPS python src/stage3_analysis/umt_bridge_phase3.py
    "
