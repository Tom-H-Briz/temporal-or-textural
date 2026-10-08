#!/bin/bash
#SBATCH --job-name=tot_k400_sl_manifest
#SBATCH --output=build_k400_sl_manifest_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00

# One-off, CPU-only (mirrors make_probe_config.sh) — builds the static
# k400_manifest_SL_subset.json (Step 0, 03/08 CC brief). Run this ONCE, before
# run_position_lock_vm_kinetics.sh / dfa_mass_delta_vm.py --dataset kinetics400
# / ablation_cross_l5_l7.py --dataset kinetics400 — not baked into those jobs,
# since the whole point of a static manifest is that it's built once and then
# reused unchanged, not silently re-derived (and potentially drifting) on
# every downstream run. Needs k400_sl_class_mapping.csv already present on
# Isambard at outputs/Laura_SL/ (same precondition run_position_lock_vm_kinetics.sh
# already documents).

source $HOME/.tokens

export KINETICS_LABELS_CSV="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet pandas &&
        cd $HOME/temporal-or-textural &&
        python notebooks/build_k400_sl_manifest.py
    "
