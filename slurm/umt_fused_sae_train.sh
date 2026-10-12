#!/bin/bash
#SBATCH --job-name=tot_umt_fused_sae
#SBATCH --output=umt_fused_sae_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=9,10

# Fused-text SAEs for UMT (text-encoder layers 9 & 10), all-174 manifest (6,705 clips
# x 174 templates, real tokens only), k64 x8 alpha 0.03 aux, 7 epochs, job label 7ep.
# Chain per array task: manifest regen if missing -> dim_mean (1,000 clips) if missing
# -> train (self-resuming on time-limit kill) -> eval gates (VTM splice + verb
# signature). Vision-side SAEs are untouched. See train_sae_umt_fused.py docstring.

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MANIFEST_PATH="$HOME/temporal-or-textural/outputs/manifests/manifest_ssv2_all174_cap40.json"
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
export SPLICE_CAP=300

REPO="$HOME/temporal-or-textural"
DIM="$REPO/outputs/sae/umt_ssv2_fusedtext_layer${SLURM_ARRAY_TASK_ID}_dim_mean.pt"
_CKPT="$REPO/outputs/sae/sae_umt_ssv2_fusedtext_k64_x8_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    echo "RESUME_FROM=$_CKPT"
    export RESUME_FROM="$_CKPT"
fi

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd $REPO &&
        if [ ! -f '$MANIFEST_PATH' ]; then
            echo 'regenerating all-174 manifest' &&
            python notebooks/build_all174_manifest.py
        fi &&
        if [ ! -f '$DIM' ]; then
            MODE=dim_mean DIM_MEAN_CLIPS=1000 python src/stage2_sae/train_sae_umt_fused.py
        fi &&
        MODE=train python src/stage2_sae/train_sae_umt_fused.py &&
        MODE=eval python src/stage2_sae/train_sae_umt_fused.py
    "
