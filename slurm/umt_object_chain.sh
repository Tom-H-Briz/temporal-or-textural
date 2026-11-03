#!/bin/bash
#SBATCH --job-name=tot_umt_obj_chain
#SBATCH --output=umt_object_chain_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# UMT object pipeline at one vision layer (03/11/26), end to end:
#   umt_object_contrast.py  MODE=all   sets (shared, once) -> extract -> select -> controls
#   umt_object_ablation.py  FROM_CONTRAST=1  ablate (spliced, R + C + blank video) -> summarise
# Outputs -> outputs/analysis/umt_object_contrast/sets.csv (shared),
#            outputs/analysis/umt_object_contrast_L{L}/, outputs/analysis/umt_object_ablation_L{L}_splice/
# STAGE=sample writes sets.csv only (run once first, so the layer jobs never race on it):
#   J=$(sbatch --parsable --export=ALL,STAGE=sample slurm/umt_object_chain.sh)
#   for L in 5 7 9; do sbatch --dependency=afterok:$J --export=ALL,LAYER=$L slurm/umt_object_chain.sh; done
# TASK=pair (opposite pairs, key word masked; R + C + Rev ablation) -> umt_pair_contrast* / umt_pair_ablation_L{L}_splice:
#   J=$(sbatch --parsable --export=ALL,TASK=pair,STAGE=sample slurm/umt_object_chain.sh)
#   for L in 5 7 9; do sbatch --dependency=afterok:$J --export=ALL,TASK=pair,LAYER=$L slurm/umt_object_chain.sh; done

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export LAYER=${LAYER:-9} STAGE=${STAGE:-all} SPLICE=1 MAX_CLIPS=0
echo "LAYER=$LAYER STAGE=$STAGE"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
if [ "$STAGE" = sample ]; then
    RUN="MODE=sample python src/stage3_analysis/umt_object_contrast.py"
else
    RUN="MODE=all python src/stage3_analysis/umt_object_contrast.py &&
         FROM_CONTRAST=1 MODE=ablate python src/stage3_analysis/umt_object_ablation.py &&
         FROM_CONTRAST=1 MODE=summarise python src/stage3_analysis/umt_object_ablation.py"
fi

apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural && unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
    $RUN
"
