#!/bin/bash
#SBATCH --job-name=tot_umt25m_parity
#SBATCH --output=umt25m_parity_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Parity check (umt25m_brief_241126.md §2, agent B): official UMT class (xbert, transformers
# 4.24) vs notebooks/umt25m_wrapper.py on ~20 MSRVTT clips. xbert needs transformers 4.24,
# which the pytorch 25.05 SIF does not provide (5.5 breaks it) - so a 4.24 venv is created
# on first run (rerunnable: skipped if present). N_CLIPS env overrides the default 20.
# Smoke:  sbatch --export=ALL,N_CLIPS=5 slurm/umt25m_parity.sh
# Full:   sbatch slurm/umt25m_parity.sh

source $HOME/.tokens
export MSRVTT_ANNO="$SCRATCHDIR/data/msrvtt/anno/msrvtt_ret_test1k.json"
export MSRVTT_VIDEO_DIR="$SCRATCHDIR/data/msrvtt"
export UMT25M_CKPT="$HOME/temporal-or-textural/models/umt_ckpts/b16_25m.pth"
[ -f "$UMT25M_CKPT" ] || { echo "MISSING $UMT25M_CKPT"; exit 1; }
[ -f "$MSRVTT_ANNO" ] || { echo "MISSING $MSRVTT_ANNO"; exit 1; }
find "$MSRVTT_VIDEO_DIR" -name '*.mp4' -print -quit | grep -q . \
    || { echo "NO mp4 under $MSRVTT_VIDEO_DIR"; exit 1; }
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
VENV24="$HOME/umt24_venv"

apptainer exec \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        if [ ! -f $VENV24/bin/python ]; then
            python3 -m venv $VENV24 &&
            $VENV24/bin/pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu &&
            $VENV24/bin/pip install --quiet \"transformers==4.24.0\" av einops pandas numpy tqdm \"timm==0.4.12\" easydict huggingface_hub
        fi &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        cd \$HOME/temporal-or-textural &&
        $VENV24/bin/python notebooks/umt25m_parity_check.py
    "
