#!/bin/bash
#SBATCH --job-name=tot_umt_mlm_grp_abl
#SBATCH --output=umt_mlm_group_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# UMT MLM group ablation (src/stage3_analysis/umt_mlm_group_ablation.py), brief
# 201026: ablating gated / flip / no-change feature groups (5 per pair) at the L9
# vision SAE, against the refit MLM head's masked pair word and the caption's other
# words, on g34 clips of the three clean pairs, R and Rev, with mass-matched control
# sets and singles. Outputs -> outputs/analysis/umt_mlm_group_ablation/.
# MODE=groups|check|ablate|summarise|all (a failed stage can be resubmitted alone).
# groups must run first (CPU, inside this script's allocation under MODE=all is fine):
# check/ablate read its groups.csv, built from the contrast run's sparse/ files.
# MAX_CLIPS caps the clip set for CPU checks (outputs go to *_smoke, never the real run).
# Full:                sbatch slurm/umt_mlm_group_ablation.sh
# Stage alone:         sbatch --export=ALL,MODE=ablate slurm/umt_mlm_group_ablation.sh

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
export MAX_CLIPS=${MAX_CLIPS:-0}
export MODE=${MODE:-all}
echo "MODE=$MODE MAX_CLIPS=$MAX_CLIPS"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python src/stage3_analysis/umt_mlm_group_ablation.py \$MODE
    "
