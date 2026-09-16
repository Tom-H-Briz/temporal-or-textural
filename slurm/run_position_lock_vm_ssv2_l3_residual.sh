#!/bin/bash
#SBATCH --job-name=tot_pos_lock_vm_ssv2_l3_residual
#SBATCH --output=position_lock_vm_ssv2_l3_residual_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=3

# Residual-stream variant of run_position_lock_vm_ssv2_l3.sh: --source residual
# identity-splices layer L3's raw activation instead of a trained SAE (see
# ResidualDFAEngine in dfa_engine.py) — no SAE checkpoint needed, dict_size is
# VideoMAE's hidden_dim (768). Path fixed to dfa_extraction/ (03/09/26 move;
# the file no longer exists at the old src/stage3_analysis/ location this
# script's template pointed at).

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
        python src/stage3_analysis/dfa_extraction/position_lock_extraction.py --model videomae --layer $SLURM_ARRAY_TASK_ID --source residual
    "
