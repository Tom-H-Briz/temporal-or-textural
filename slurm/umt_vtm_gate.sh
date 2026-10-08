#!/bin/bash
#SBATCH --job-name=tot_umt_vtm_gate
#SBATCH --output=umt_vtm_gate_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# UMT VTM gate (src/stage3_analysis/umt_vtm_gate.py): VTC logits + VTM scores for all 174 templates,
# full SSv2 val R, then aggregation. Same env as umt_ssv2_perturb_accuracy.sh (+ scipy for McNemar);
# VTM fusion runs in plain torch, so no old transformers is needed.
# Smoke:  sbatch --export=ALL,MAX_CLIPS=200 slurm/umt_vtm_gate.sh   (-> outputs/analysis/umt_vtm_gate_smoke/)
# Full:   sbatch slurm/umt_vtm_gate.sh

source $HOME/.tokens
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python src/stage3_analysis/umt_vtm_gate.py all
    "
