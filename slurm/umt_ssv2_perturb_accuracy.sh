#!/bin/bash
#SBATCH --job-name=tot_umt_ssv2_perturb
#SBATCH --output=umt_ssv2_perturb_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=03:00:00

# UMT-B (SSv2-template retrieval ckpt) per-class accuracy, FULL SSv2 val (24,777),
# conditions R / C / A (TF convention). VM-SSv2's R/A/C1 run took ~10 min/condition;
# 3h is first-run headroom for a new backbone.
# Needs on Isambard (gitignored, rsync'd once): models/umt_ckpts/ret_ssv2_tpl_b16_25m.pth,
# models/umt_ckpts/ssv2_template_text_emb.npy, and the repo clone at models/unmasked_teacher.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

# timm==0.4.12 + easydict: UMT's own vit.py (models/unmasked_teacher) needs them.
# transformers stays 5.5.0 — the vision path never imports it (text emb is cached).
apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/perturb_accuracy_umt.py
    "
