#!/bin/bash
#SBATCH --job-name=tot_tf_k400_dim_mean
#SBATCH --output=tf_k400_dim_mean_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --array=5,7,9

# Job 1 — dim_mean, computed from scratch under the TF-K400 sampler (8 frames,
# 64-frame centre window). 2,000 clips = profile_activations default, matching
# the VM-K400 sweep. Output: outputs/sae/tf_kinetics400_layer{5,7,9}_dim_mean.pt
# (bare tensor — train_sae loads it weights_only=True).
# Prereq: Job 0 passed (dependency wired in submit_tf_k400_chain.sh).

source $HOME/.tokens

export MODEL_NAME=timesformer
export DATASET_NAME=kinetics400
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/profile_activations.py
    "
