#!/bin/bash
#SBATCH --job-name=tot_umt_frozen_pe
#SBATCH --output=umt_frozen_pe_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Frozen-PE ablation of the order-sensitive / insensitive groups (19/11/26), spliced, L5 / L7 / L9: DFA, activation,
# logits full vs frozen PE + group-patched logits -> outputs/analysis/umt_frozen_pe_groups/L{5,7,9}.npz (~0.4 GB each).
#   sbatch slurm/umt_frozen_pe_groups.sh            smoke: sbatch --time=00:20:00 --export=ALL,MAX_CLIPS=4 ...
source $HOME/.tokens   # exports HF_TOKEN
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export MAX_CLIPS=${MAX_CLIPS:-0} NUM_WORKERS=${NUM_WORKERS:-4}
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural &&
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
    python -u src/stage3_analysis/umt_frozen_pe_groups.py
"
