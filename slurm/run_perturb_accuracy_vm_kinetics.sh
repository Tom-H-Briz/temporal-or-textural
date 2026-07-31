#!/bin/bash
#SBATCH --job-name=tot_perturb_acc_vm_kinetics
#SBATCH --output=run_perturb_accuracy_vm_kinetics_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# Conditions A/C1 only. No VIDEO_DIR/KINETICS_LABELS_CSV override — DATASET_REGISTRY
# ["kinetics400"] and CFG's default resolve paths, same convention as
# train_sae_vm_kinetics.sh.

source $HOME/.tokens

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/perturb_accuracy_vm_kinetics.py
    "
