#!/bin/bash
#SBATCH --job-name=tot_tf_k400_dfa
#SBATCH --output=tf_k400_dfa_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=5,7,9

# Job 4b — per-layer DFA stack over the K400 SL manifest clips:
#   1. dfa_mass_delta_vm.py --model timesformer  -> per-clip R/C/A parquet
#      (two-stage: raw per-clip rows persisted before any aggregation)
#   2. position_lock_extraction.py               -> DFA + raw-z position stats
#   3. cumulative_mass_diagnostic_tf.py          -> reliability parquet
#      (the four-bucket's max_mean_ratio source, mirroring ssv2_tf)
# 12h per task: K400 clips decode ~2x longer than SSv2 (VM-K400 precedent).
# Prereqs: Job 2 SAEs (dependency-wired), VIDEO_DIR manifest clips on scratch.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset"
export DATASET_NAME=kinetics400
L=$SLURM_ARRAY_TASK_ID

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py \
            --model timesformer --dataset kinetics400 --layer $L --job-label 7ep --sae-k 64 &&
        python src/stage3_analysis/dfa_extraction/position_lock_extraction.py \
            --model timesformer --dataset kinetics400 --layer $L &&
        DATASET_NAME=kinetics400 SAE_LAYER=$L \
            python src/stage3_analysis/dfa_extraction/cumulative_mass_diagnostic_tf.py
    "
