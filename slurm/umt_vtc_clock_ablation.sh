#!/bin/bash
#SBATCH --job-name=tot_umt_vtc_clock
#SBATCH --output=umt_vtc_clock_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# UMT L9 clock ablation under the VTC head, SL subset (src/stage3_analysis/umt_vtc_clock_ablation.py,
# brief 211026): 7 clock features + hand singles 4788/6127, mass-matched group controls, all
# 174 logits per clip x arm. Outputs -> outputs/analysis/umt_vtc_clock_ablation/.
# MODE=mass|check|ablate|summarise|compare|all (check needs mass first; compare is post-fix).
# Full:        sbatch slurm/umt_vtc_clock_ablation.sh
# One stage:   sbatch --export=ALL,MODE=ablate slurm/umt_vtc_clock_ablation.sh
# Smoke (5 clips, *_smoke dir): sbatch --export=ALL,MODE=all,MAX_CLIPS=5 slurm/umt_vtc_clock_ablation.sh

source $HOME/.tokens   # exports HF_TOKEN
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export MAX_CLIPS=${MAX_CLIPS:-0}
export MODE=${MODE:-all}
echo "MODE=$MODE MAX_CLIPS=$MAX_CLIPS"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python src/stage3_analysis/umt_vtc_clock_ablation.py \$MODE
    "
