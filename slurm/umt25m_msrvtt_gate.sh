#!/bin/bash
#SBATCH --job-name=tot_umt25m_gate
#SBATCH --output=umt25m_msrvtt_gate_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# MSRVTT zero-shot gate for UMT 25M (notebooks/umt25m_msrvtt_eval.py; umt25m_brief_241126.md §2).
# Prereqs: msrvtt_download.sh finished (videos under $SCRATCHDIR/data/msrvtt) and the
# user-fetched msrvtt_ret_test1k.json under $SCRATCHDIR/data/msrvtt/anno/.
# Smoke:  sbatch --export=ALL,LIMIT=50 slurm/umt25m_msrvtt_gate.sh   (-> outputs/analysis/umt25m_msrvtt_gate_smoke/)
# Full:   sbatch slurm/umt25m_msrvtt_gate.sh

source $HOME/.tokens
export MSRVTT_ANNO="$SCRATCHDIR/data/msrvtt/anno/msrvtt_ret_test1k.json"
export MSRVTT_VIDEO_DIR="$SCRATCHDIR/data/msrvtt"
export UMT25M_CKPT="$HOME/temporal-or-textural/models/umt_ckpts/b16_25m.pth"
[ -f "$UMT25M_CKPT" ] || { echo "MISSING $UMT25M_CKPT"; exit 1; }
[ -f "$MSRVTT_ANNO" ] || { echo "MISSING $MSRVTT_ANNO"; exit 1; }
find "$MSRVTT_VIDEO_DIR" -name '*.mp4' -print -quit | grep -q . \
    || { echo "NO mp4 under $MSRVTT_VIDEO_DIR"; exit 1; }
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python notebooks/umt25m_msrvtt_eval.py
    "
