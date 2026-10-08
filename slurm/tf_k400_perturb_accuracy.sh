#!/bin/bash
#SBATCH --job-name=tot_tf_k400_perturb
#SBATCH --output=tf_k400_perturb_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Job 4a — TF-K400 per-class accuracy, FULL matched val set (~19-20k clips),
# conditions R / C / A. This is the class-pool source for the four-bucket run
# (>=40% R-acc eligibility, mirroring TF-SSv2's clip_shuffle_disruption entry)
# and the condition-accuracy mirror of TF-SSv2's per_class_accuracy_TF*.csv.
# 3 conditions x ~20k clips at TF's 8 frames — 12h ceiling, same budget class
# as the VM-K400 perturb run.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/perturb_accuracy_tf_kinetics.py
    "
