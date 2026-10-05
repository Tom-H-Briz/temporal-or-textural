#!/bin/bash
#SBATCH --job-name=tot_all174_dfa_rc
#SBATCH --output=all174_dfa_rc_%A_%a.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --array=0-8

# Template-word-overlap run: R + shuffle DFA (no A) over ALL 174 SSv2 classes, <=40 val clips
# per class (6,705 clips; manifest md5 fcd47302456c691b9c5528344f488c48), R-correct gated.
# Tasks 0-8 = {umt, videomae, timesformer} x {5, 7, 9}. Shuffle per backbone: VM C1, TF/UMT C.
# Output: outputs/analysis/dfa_mass_delta_{umt,vm_c1,tf}/..._ssv2_all174_l{L}_... .parquet
# 4h: ~6.7k clips x 2 passes vs the SL run's 5.8k x 3 (VM-SSv2 SL limit was 2h).
# Prereq (once, login node): VALIDATION_PATH=$HOME/labels/validation.json python3 notebooks/build_all174_manifest.py

MODELS=(umt umt umt videomae videomae videomae timesformer timesformer timesformer)
LAYERS=(5 7 9 5 7 9 5 7 9)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
L=${LAYERS[$SLURM_ARRAY_TASK_ID]}
MANIFEST="$HOME/temporal-or-textural/outputs/manifests/manifest_ssv2_all174_cap40.json"
[ -f "$MANIFEST" ] || { echo "MISSING MANIFEST: $MANIFEST (run the builder first)"; exit 1; }
echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL layer=$L"

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
        pip install --quiet av einops pandas pyarrow matplotlib \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        python src/stage3_analysis/dfa_extraction/dfa_mass_delta_vm.py \
            --model $MODEL --dataset ssv2 --layer $L --job-label 7ep --sae-k 64 \
            --manifest $MANIFEST --out-tag all174 --skip-a
    "
