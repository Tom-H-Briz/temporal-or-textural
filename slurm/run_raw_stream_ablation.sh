#!/bin/bash
#SBATCH --job-name=tot_raw_stream_abl
#SBATCH --output=raw_stream_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# Full 4,450-clip population — n=500 resolved the main effect (0.228 raw,
# p=4e-11) but not the temporal/static interaction (gap +0.018, CI spanning
# zero) and gives only ~15 clips/class, so no per-class breakdown is possible.
# That is what this run is for; see 220926_workbook.md finding 5.
#
# 40 forward passes per clip: raw + recon baselines; subtract and mean-project
# (Dobrzeniecka et al. 2025) x raw/recon for the 7 singles, all7 and
# rand_dict7; mean-project x raw/recon for rand_iso7. Zero projection dropped
# 24/09 (no precedent; adds an off-distribution shift). Measured 5.7s/clip on
# local MPS. All passes reuse one clip decode, which is CPU-bound and
# dominates; GPU should land near 2s/clip (~2.5h), so 4h is comfortable.
#
# Controls are drawn once per clip and shared by every operation, so
# subtraction and mean projection face the same random features. rand_dict7 is
# mass-matched per clip (7 non-scaffold features from the clip's top-50 by
# activation mass, within 10% of the scaffold's total) — unmatched random
# features carry ~4% of the scaffold's mass and would make a trivially weak
# subtraction control. Each row records control_mass_ratio for audit.
#
# Overwrites raw_stream_ablation_l5_n4450.parquet. The old rand_dict7 figures
# (unmatched, projection only) are NOT comparable to the new ones.
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
