#!/bin/bash
#SBATCH --job-name=tot_ablation_cross_l5_l7_kinetics
#SBATCH --output=ablation_cross_l5_l7_kinetics_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# K400 equivalent of run_ablation_cross_l5_l7.sh. Both L5 and L7 SAEs spliced
# simultaneously, ablating the combined K400 member set (L5's confirmed 10 +
# L7's confirmed 6, from ablation_targets.py's registry — no hand-edit needed,
# 04/08 registry fix) — R condition only. Standalone script
# (ablation_cross_l5_l7.py), not run_ablation.py — DFAEngine only splices one
# layer at a time. Source: k400_manifest_SL_subset.json (~3186 clips),
# R-correctness recomputed fresh under the dual-spliced baseline, independent
# of run_ablation.py's per-layer R-correct rosters (splicing two SAEs at once
# can shift borderline clips) and of the mass-delta parquets (this script
# doesn't read them at all).
# Time budget doubled vs the SSv2 script, same K400-clips-run-longer reasoning
# as the other K400 launchers.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/ablation_cross_l5_l7.py --dataset kinetics400
    "
