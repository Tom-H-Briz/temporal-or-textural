#!/bin/bash
#SBATCH --job-name=tot_umt_obj_abl
#SBATCH --output=umt_object_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# UMT object ablation (src/stage3_analysis/umt_object_ablation.py, 03/11/26): mask the
# object noun (box / pen / cup) in every val caption that has it once, ablate the L9
# object features (box #5483, cup #6011 / #5880 / pair, pen #2428) and mass-matched
# controls, spliced, under R and C, + a blank-video pass -> refit MLM head log p(noun).
# Outputs -> outputs/analysis/umt_object_ablation_splice/per_row.parquet.
# Full: sbatch slurm/umt_object_ablation.sh   (SPLICE=0 for the raw-stream confirmation)

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MAX_CLIPS=${MAX_CLIPS:-0}
export SPLICE=${SPLICE:-1}
echo "SPLICE=$SPLICE MAX_CLIPS=$MAX_CLIPS"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural && unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
    python src/stage3_analysis/umt_object_ablation.py
"
