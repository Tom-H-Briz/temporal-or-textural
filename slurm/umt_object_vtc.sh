#!/bin/bash
#SBATCH --job-name=tot_umt_obj_vtc
#SBATCH --output=umt_object_vtc_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00

# Object features vs the VTC class head (src/stage3_analysis/umt_object_vtc.py, 05/11/26): the layer's
# object edits (singles, groups, mass-matched controls; spliced, R) on every object test clip.
# Needs that layer's umt_object_chain.sh outputs. Outputs -> outputs/analysis/umt_object_vtc_L{L}_splice/
# per_clip.parquet (summary: MODE=summarise locally). All three: for L in 5 7 9; do sbatch --export=ALL,LAYER=$L slurm/umt_object_vtc.sh; done
# Verb pairs (-> umt_pair_vtc_L{L}_splice, within-pair accuracy too): add TASK=pair to --export

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos" LABELS_PATH="$HOME/labels/labels.json" VALIDATION_PATH="$HOME/labels/validation.json"
export LAYER=${LAYER:-9} TASK=${TASK:-object} MODE=ablate MAX_CLIPS=0
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural && unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE && python src/stage3_analysis/umt_object_vtc.py
"
