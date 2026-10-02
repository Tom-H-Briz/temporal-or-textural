#!/bin/bash
#SBATCH --job-name=tot_tf_k400_four_bucket
#SBATCH --output=tf_k400_four_bucket_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# Job 5 — four-bucket (clip_shuffle_disruption --only k400_tf: eligibility
# >=40% R-acc from Job 4a's full-val per-class file, top-10, 5% noise band,
# noise->sign_flip->decrease/increase locked order, reliability parquet from
# Job 4b's cumulative-mass diagnostic) + report assembly + verification counts
# (3/3/3/3; exits non-zero on shortfall). No interpretation, numbers as found.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow statsmodels \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python src/stage3_analysis/clip_shuffle_disruption.py --only k400_tf &&
        python scripts/build_tf_k400_reports.py
    "
