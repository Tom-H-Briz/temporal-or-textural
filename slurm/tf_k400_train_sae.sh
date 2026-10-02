#!/bin/bash
#SBATCH --job-name=tot_tf_k400_train_sae
#SBATCH --output=tf_k400_train_sae_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=5,7,9

# Job 2 — TF-K400 SAEs, config identical to the canonical VM-K400 k64/x8 set:
# k=64, expansion=8, alpha=0.03, aux loss, 7 epochs, job label 7ep (read back
# from the VM-K400 checkpoints' own hyperparameters, recorded in the report).
# SAE_VAL_FRACTION=0.2 + auto-RESUME_FROM copied from train_sae_vm_kinetics_
# l5_l7_l9.sh: K400 val (19,881 clips) < train_clips=20,000, so the legacy
# count-based split would leave the val loader EMPTY (no health metrics);
# the 20% split also persists the held-out clip list. Fail-loud dim_mean guard
# already in train_sae.py (FileNotFoundError). Health metrics go to wandb
# (R^2, L0, dead counts per epoch; near-dead via the firing-rate histogram).

source $HOME/.tokens   # exports HF_TOKEN, WANDB_API_KEY

export MODEL_NAME=timesformer
export DATASET_NAME=kinetics400
export SAE_LAYER=$SLURM_ARRAY_TASK_ID
export SAE_K=64
export SAE_EXPANSION=8
export SAE_ALPHA=0.03
export SAE_LOSS_FN=aux
export SAE_EPOCHS=7
export SAE_JOB_LABEL=7ep
export SAE_VAL_FRACTION=0.2
export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"

# Auto-resume: safe to resubmit this script as-is if a task gets killed
# (rolling-latest checkpoint pattern, same as the VM-K400 sweep).
_CKPT="$HOME/temporal-or-textural/outputs/sae/sae_tf_kinetics400_k64_x8_l${SLURM_ARRAY_TASK_ID}_job7ep.pt"
if [ -f "$_CKPT" ]; then
    echo "RESUME_FROM=$_CKPT"
    export RESUME_FROM="$_CKPT"
fi

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops wandb pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd \$HOME/temporal-or-textural &&
        python src/stage2_sae/train_sae.py
    "
