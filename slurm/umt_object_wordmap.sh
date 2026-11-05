#!/bin/bash
#SBATCH --job-name=tot_umt_obj_wordmap
#SBATCH --output=umt_object_wordmap_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# UMT object word maps (src/stage3_analysis/umt_object_wordmap.py, 03/11/26): every word of each
# held-out test caption masked alone, scored under the layer's object-feature edits (singles,
# groups, mass-matched controls; spliced, R). Needs that layer's umt_object_chain.sh outputs.
# Outputs -> outputs/analysis/umt_object_wordmap_L{L}/per_word.parquet (plots: MODE=plot locally).
# All three layers:  for L in 5 7 9; do sbatch --export=ALL,LAYER=$L slurm/umt_object_wordmap.sh; done
# Verb pairs (group edits only, -> umt_pair_wordmap_L{L}): add TASK=pair to --export
export TASK=${TASK:-object}

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export LAYER=${LAYER:-9} MODE=run MAX_CLIPS=0
echo "LAYER=$LAYER TASK=$TASK"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural && unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
    python src/stage3_analysis/umt_object_wordmap.py
"
