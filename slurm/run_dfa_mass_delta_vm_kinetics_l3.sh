#!/bin/bash
#SBATCH --job-name=tot_dfa_mass_delta_vm_kinetics_l3
#SBATCH --output=dfa_mass_delta_vm_kinetics_l3_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --array=3

# L3 extension of run_dfa_mass_delta_vm_kinetics_l5_l7.sh — same job7ep/k64
# config, one more layer point. Prereq: train_sae_vm_kinetics_l3.sh completed
# (writes sae_vmae_kinetics400_k64_x8_l3_job7ep_best.pt). Time budget matches
# the L5/L7 K400 script — same decode-cost reasoning, layer choice doesn't
# change per-clip forward/backward pass cost.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_mass_delta_vm.py --dataset kinetics400 --layer \$SLURM_ARRAY_TASK_ID --job-label 7ep --sae-k 64
    "
