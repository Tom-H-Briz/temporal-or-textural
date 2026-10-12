#!/bin/bash
#SBATCH --job-name=tot_umt_bridge
#SBATCH --output=umt_bridge_%A.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# UMT vision->text bridge, Phase 1 (brief dfa_attention_bridge_brief_121026.md):
# G1 gradient-path checks -> G0 174-way raw-vs-spliced (fused SAEs L9/L10, 805 held-out)
# -> T1 targets on real clips (8 usable pairs, all 3,647 clips; D1 rides the pass).
# MAX_CLIPS>0 gives a smoke run. TF32 must stay OFF (xbert self-check tolerance).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MANIFEST_PATH="$HOME/temporal-or-textural/outputs/manifests/manifest_ssv2_all174_cap40.json"
export MODE=all
export MAX_CLIPS=${MAX_CLIPS:-0}

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd $HOME/temporal-or-textural &&
        if [ ! -f '$MANIFEST_PATH' ]; then
            echo 'regenerating all-174 manifest' &&
            python notebooks/build_all174_manifest.py
        fi &&
        MODE=all python src/stage3_analysis/umt_vision_text_bridge.py
    "
