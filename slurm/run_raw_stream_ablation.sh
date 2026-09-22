#!/bin/bash
#SBATCH --job-name=tot_raw_stream_abl
#SBATCH --output=raw_stream_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00

# Full 4,450-clip population — n=500 resolved the main effect (0.228 raw,
# p=4e-11) but not the temporal/static interaction (gap +0.018, CI spanning
# zero) and gives only ~15 clips/class, so no per-class breakdown is possible.
# That is what this run is for; see 220926_workbook.md finding 5.
#
# 34 forward passes per clip (raw + recon baselines, then subtract/project x
# raw/recon x 7 singles + the all7 group) — ~17x run_ablation_l5.sh's 2-pass
# workload, hence 6h not 2h. Measured 5.0-5.5s/clip on local MPS, but the 34
# passes reuse one decode, so GPU should land nearer 2s/clip (~2.5h) — decode
# is CPU-bound and will not scale, which is what the headroom covers.
#
# --no-activations: the per-clip h_raw/z dump is 3.5MB/clip = ~15.6GB over the
# full set, and is only needed for fingerprint work (already done at n=100).
# Per-class logit deltas do not need it.
#
# Not chained into raw_stream_ablation_summary.py — same convention as
# run_ablation_l5.sh, run the summary as a separate step once this lands.
# NOTE: the summary script imports scipy, which is not in the pip line below.

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
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/raw_stream_feature_ablation.py --n-clips 0 --no-activations
    "
