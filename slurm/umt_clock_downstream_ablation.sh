#!/bin/bash
#SBATCH --job-name=tot_umt_clockdown
#SBATCH --output=umt_clockdown_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# UMT clock ablation with downstream capture (src/stage3_analysis/umt_clock_downstream_ablation.py, 10/11/26):
# spliced ablation at LAYER (5 or 7) of the positional arms + clock (ramps + near-miss); per-clip x frame profiles of
# all latents at every later SAE layer (7, 9), block LAYER+1 attention frame x frame, baseline residual norms + comb map.
# Clips: all val clips of 5 verb pairs, R + Rev. Outputs (memmapped .npy, resumable) ->
# outputs/analysis/umt_clock_downstream_L{LAYER}/  (~8.5 GB for LAYER=5, ~4.3 GB for LAYER=7).
#   for L in 5 7; do sbatch --export=ALL,LAYER=$L slurm/umt_clock_downstream_ablation.sh; done
# Smoke (2 clips, own dir): sbatch --time=00:20:00 --export=ALL,LAYER=5,MAX_CLIPS=2,OUT_DIR=outputs/analysis/umt_clock_downstream_smoke slurm/umt_clock_downstream_ablation.sh
# Raw-stream ablation + attribution (what LAYER latents build the key features): add SPLICE=0 (ATTR_BATCH=8 default).
# Resume after a crash: resubmit the same command; finished clips are skipped (done_{R,Rev}.npy).
# VideoMAE (11/11/26, raw L5 only; + pairs 8/9, + baseline-only C1 residual pass) -> outputs/analysis/vm_clock_downstream_L5_raw/:
#   sbatch --export=ALL,MODEL=videomae,LAYER=5,SPLICE=0 slurm/umt_clock_downstream_ablation.sh

source $HOME/.tokens   # exports HF_TOKEN
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export MODEL=${MODEL:-umt}   # umt | videomae
export LAYER=${LAYER:-5}
export SPLICE=${SPLICE:-1}   # 0 = raw-stream subtraction (dose 1) -> outputs/analysis/{umt,vm}_clock_downstream_L{LAYER}_raw/
export MAX_CLIPS=${MAX_CLIPS:-0}
export NUM_WORKERS=${NUM_WORKERS:-4}
[ -n "$OUT_DIR" ] && export OUT_DIR
echo "MODEL=$MODEL LAYER=$LAYER SPLICE=$SPLICE MAX_CLIPS=$MAX_CLIPS NUM_WORKERS=$NUM_WORKERS OUT_DIR=${OUT_DIR:-default}"
ls $(dirname $LABELS_PATH)/validation.json || { echo "validation.json missing next to labels.json (set VAL_JSON)"; exit 1; }
df -h $HOME /dev/shm
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        python -u src/stage3_analysis/umt_clock_downstream_ablation.py
    "
