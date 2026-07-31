#!/bin/bash
#SBATCH --job-name=tot_dfa_tubelet_vm_kinetics
#SBATCH --output=dfa_per_tubelet_mass_vm_kinetics_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=5,7,9

# K400 arm — 64 SL-matched classes (outputs/Laura_SL/k400_sl_class_mapping.csv),
# not yet filtered to an eligible subset (per-class accuracy >= 40% gate is separate
# follow-up, pending perturb_accuracy_vm_kinetics.py's R-condition results). Needs
# k400_sl_class_mapping.csv present on Isambard — outputs/ is gitignored, so either
# rerun notebooks/k400_sl_mapping.py here (cheap, no GPU) or sync the CSV over.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_per_tubelet_mass.py --model videomae --dataset kinetics400 --layer $SLURM_ARRAY_TASK_ID
    "
