#!/bin/bash
#SBATCH --job-name=tot_umt_ssv2_four_bucket
#SBATCH --output=umt_ssv2_four_bucket_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Four-bucket shuffle behaviour (clip_shuffle_disruption --only umt -> umt_l5/l7/l9):
# eligibility >=40% R-acc from perturb_accuracy_umt.py's full-val CSV, top-10, 5% band,
# noise->sign_flip->decrease/increase locked order; reliability ratio computed inline
# (VM precedent). Output: outputs/analysis/shuffle_reduction_composition/umt_l*_*.csv

source $HOME/.tokens

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow statsmodels \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python src/stage3_analysis/clip_shuffle_disruption.py --only umt
    "
