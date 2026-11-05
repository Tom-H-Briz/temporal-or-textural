#!/bin/bash
#SBATCH --job-name=tot_umt_posfeat
#SBATCH --output=umt_posfeat_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00

# UMT positional-feature ablation + frame profiles at L5/L7/L9
# (src/stage3_analysis/umt_layer_positional_ablation.py, brief 051126): the selection arms
# (ramp / nonramp / positional / near_flat / all) (no controls), all 174 logits per
# clip x arm, per-clip frame profiles for R/C/Rev. Outputs ->
# outputs/analysis/umt_layer_positional_ablation_L{LAYER}_splice/.
# MODE=mass|check|ablate|profiles|summarise (mass before ablate/check; profiles before summarise).
# One job per layer:  sbatch --export=ALL,LAYER=5 slurm/umt_layer_positional_ablation.sh
# One stage:          sbatch --export=ALL,LAYER=5,MODE=mass slurm/umt_layer_positional_ablation.sh
# Smoke (5 clips, *_smoke dir): sbatch --export=ALL,LAYER=5,MODE=mass,MAX_CLIPS=5 slurm/umt_layer_positional_ablation.sh

source $HOME/.tokens   # exports HF_TOKEN
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export LAYER=${LAYER:-5}
export MODE=${MODE:-mass}
export MAX_CLIPS=${MAX_CLIPS:-0}
export NUM_WORKERS=${NUM_WORKERS:-4}
export SPLICE=${SPLICE:-1}
echo "LAYER=$LAYER MODE=$MODE MAX_CLIPS=$MAX_CLIPS NUM_WORKERS=$NUM_WORKERS SPLICE=$SPLICE"
df -h /dev/shm                                   # loader workers share decoded clips via /dev/shm
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python -u src/stage3_analysis/umt_layer_positional_ablation.py \$MODE
    "
