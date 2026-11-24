#!/bin/bash
#SBATCH --job-name=tot_tf_pe_char
#SBATCH --output=tf_pe_char_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=02:00:00

# TimeSformer dictionary-wide PE characterisation at L5 / L7 / L9 (24/11/26): blank (full / frozen time table) + UMT's 696 clip
# ids as R / C / Rev -> outputs/analysis/tf_layer_pe_characterise/L{5,7,9}.npz. Synth clips run locally (SYNTH_ONLY=1).
#   sbatch slurm/tf_layer_pe_characterise.sh            smoke: sbatch --time=00:20:00 --export=ALL,MAX_CLIPS=4 ...
source $HOME/.tokens   # exports HF_TOKEN
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export MAX_CLIPS=${MAX_CLIPS:-0} NUM_WORKERS=${NUM_WORKERS:-4}
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
apptainer exec --nv --bind $HOME:$HOME --bind $SCRATCHDIR:$SCRATCHDIR $SIF bash -c "
    pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
    cd \$HOME/temporal-or-textural &&
    unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
    python -u src/stage3_analysis/tf_layer_pe_characterise.py
"
