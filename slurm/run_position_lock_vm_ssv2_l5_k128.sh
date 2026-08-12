#!/bin/bash
#SBATCH --job-name=tot_pos_lock_vm_ssv2_l5_k128
#SBATCH --output=position_lock_vm_ssv2_l5_k128_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=5

# k128/x16 L5 VM never got a post-30/07-fix position-lock run — only k64/x8
# was rerun for L3/L5/L7/L9. The job7ep checkpoint has existed since 30/07;
# this just runs the same extraction against it. Same time budget as the
# k64 L5 job — dict-size doubling only adds SAE encode/decode cost, which is
# small next to the VideoMAE forward+backward pass per clip.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export VALIDATION_PATH="$HOME/labels/validation.json"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/position_lock_extraction.py --model videomae --layer $SLURM_ARRAY_TASK_ID --sae-k 128
    "
