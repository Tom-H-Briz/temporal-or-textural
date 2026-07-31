#!/bin/bash
#SBATCH --job-name=tot_dfa_mass_delta_vm_l5_l7
#SBATCH --output=dfa_mass_delta_vm_l5_l7_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --array=5,7

# Fresh job7ep/k64 dfa_mass_delta_vm_c1 run for L5 and L7 — unblocks run_ablation.py
# (which needs its R-correct clip roster) and the 6 other scripts that read its
# signed_vec_R/C1/A vectors (select_control_features.py, per_class_feature_delta.py,
# feature_vis_vm.py, f5384_shape_analysis.py, taxonomy_group_screen.py,
# scaffold_mass_pct.py). k=64 to match the SAE config the fresh position-locked
# features (358.../5165...) were derived from — scan_dfa_mass_delta_vm_configs.sh's
# L7 entry is k=128, a different config, not what we need here.

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
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_mass_delta_vm.py --layer \$SLURM_ARRAY_TASK_ID --job-label 7ep --sae-k 64
    "
