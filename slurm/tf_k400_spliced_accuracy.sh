#!/bin/bash
#SBATCH --job-name=tot_tf_k400_spliced
#SBATCH --output=tf_k400_spliced_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=5,7,9

# Job 3 — spliced accuracy per layer, on the SAME 3,000-clip sample as Job 0
# (eval_clips_3000.json persisted by the baseline gate). Report only; no
# auto-gate — which layer(s) go to DFA is Tom's call.
# Output CSVs: outputs/spliced_accuracy_vm/spliced_accuracy_l{L}_kinetics400_
# sae_tf_kinetics400_k64_x8_l{L}_job7ep_best.csv

source $HOME/.tokens

export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b6o/tomheslin83.b6o/data/kinetics400/kinetics-dataset/val.csv"

L=$SLURM_ARRAY_TASK_ID
CKPT="outputs/sae/sae_tf_kinetics400_k64_x8_l${L}_job7ep_best.pt"
DIMMEAN="outputs/sae/tf_kinetics400_layer${L}_dim_mean.pt"
EVALCLIPS="outputs/tf_k400/eval_clips_3000.json"

# Fail loud before the GPU job if any prereq is missing (belt-and-braces with
# the in-code guards — a 2h array task shouldn't die at clip 1 of 3000).
for f in "$CKPT" "$DIMMEAN" "$EVALCLIPS"; do
    [ -f "$HOME/temporal-or-textural/$f" ] || { echo "MISSING PREREQ: $f"; exit 1; }
done

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python notebooks/spliced_accuracy_vm.py --model-name timesformer \
            --dataset-name kinetics400 --layer $L \
            --sae-checkpoint $CKPT --dim-mean-path $DIMMEAN --eval-clips $EVALCLIPS
    "
