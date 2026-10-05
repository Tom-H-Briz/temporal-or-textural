#!/bin/bash
#SBATCH --job-name=tot_umt_ssv2_dim_mean
#SBATCH --output=umt_ssv2_dim_mean_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=5,7,9

# UMT-SSv2 dim_mean under UMT's own 'middle' sampler (12 frames). 2,000 clips =
# profile_activations default, same as VM/TF. Output:
# outputs/sae/umt_ssv2_layer{5,7,9}_dim_mean.pt. 2h vs VM's 1.5h: 2352 tokens vs 1568.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MODEL_NAME=umt
export DATASET_NAME=ssv2
export SAE_LAYER=$SLURM_ARRAY_TASK_ID

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/profile_activations.py
    "
