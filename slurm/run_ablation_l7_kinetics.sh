#!/bin/bash
#SBATCH --job-name=tot_ablation_l7_kinetics
#SBATCH --output=ablation_l7_kinetics_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# K400 equivalent of run_ablation_l7.sh. BEFORE SUBMITTING: ablation_targets.py's
# TARGETS must hold L7's confirmed K400 members (single active set by design,
# see that file's docstring) — position_lock_extraction.py's 03/08 re-run
# confirmed all 6, unchanged from the old provisional set:
#   661, 1116, 1348, 4391, 4817, 4853
# Requires dfa_mass_delta_vm.py --dataset kinetics400 --layer 7 to have
# finished first (run_dfa_mass_delta_vm_kinetics_l5_l7.sh) — run_ablation.py's
# _resolve_source_parquet needs that parquet to exist.
# Time budget doubled vs the SSv2 script, same K400-clips-run-longer reasoning
# as the other K400 launchers. Not chained into ablation_summary.py — run that
# as a separate step once this output lands.

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
        python src/stage3_analysis/run_ablation.py --dataset kinetics400 --layer 7 --job-label 7ep --sae-k 64
    "
