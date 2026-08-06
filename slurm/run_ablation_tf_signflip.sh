#!/bin/bash
#SBATCH --job-name=tot_tf_signflip_abl
#SBATCH --output=tf_signflip_ablation_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# Time budget: local CPU dry-run (10 clips, run_ablation_tf.py --dry-run)
# extrapolated to 2.2h for the full 5,807-clip population; doubled since
# this is this script's first GPU run and TF's per-clip cost (7 targets x
# 2 conditions = 14 forward passes) hasn't been measured on cluster hardware.

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
        echo '=== run_ablation_tf ===' &&
        python src/stage3_analysis/run_ablation_tf.py &&
        echo '=== ablation_summary_tf_signflip ===' &&
        python src/stage3_analysis/ablation_summary_tf_signflip.py &&
        echo '=== done ==='
    "
