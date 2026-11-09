#!/bin/bash
#SBATCH --job-name=tot_vm_obj_chain
#SBATCH --output=vm_object_chain_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# VideoMAE noun / verb-pair pipeline at one layer (09/11/26): vm_object_contrast.py MODE=all (extract R/C1/Rev
# -> select -> controls) then umt_object_vtc.py MODEL=videomae (class ablation, spliced, R). Needs UMT's sets.csv
# (umt_object_chain.sh STAGE=sample). Outputs -> videomae_{task}_contrast_L{L}/, videomae_{task}_vtc_L{L}_splice/
#   for T in object pair; do for L in 5 7 9; do sbatch --export=ALL,TASK=$T,LAYER=$L slurm/vm_object_chain.sh; done; done
source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos" LABELS_PATH="$HOME/labels/labels.json" VALIDATION_PATH="$HOME/labels/validation.json"
export LAYER=${LAYER:-9} TASK=${TASK:-object} MODEL=videomae SPLICE=1 MAX_CLIPS=0
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural && unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
    MODE=all python src/stage3_analysis/vm_object_contrast.py && MODE=ablate python src/stage3_analysis/umt_object_vtc.py
"
