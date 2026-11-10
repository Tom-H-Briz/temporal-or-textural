#!/bin/bash
#SBATCH --job-name=tot_umt_posfeat
#SBATCH --output=umt_posfeat_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00

# UMT positional-feature ablation + frame profiles at L5/L7/L9
# (src/stage3_analysis/umt_layer_positional_ablation.py, brief 051126): the selection arms
# (ramp / nonramp / positional / near_flat / all) (no controls), all 174 logits per
# clip x arm, per-clip frame profiles for R/C/Rev. Outputs ->
# outputs/analysis/umt_layer_positional_ablation_L{LAYER}_splice/.
# MODE=select first (sets_L5/7/9.csv, all three layers; fails on a regression mismatch), then
# MODE=chain per layer: mass -> ablate -> profiles -> summarise (+ cross-layer table):
#   J=$(sbatch --parsable --time=00:30:00 --export=ALL,MODE=select slurm/umt_layer_positional_ablation.sh)
#   for L in 5 7 9; do sbatch --dependency=afterok:$J --export=ALL,LAYER=$L slurm/umt_layer_positional_ablation.sh; done
# MODE=rerun (6 Nov, per-clip rule 3): select -> arms (CPU, from mass_R.npy) -> ablate -> summarise,
#   for L in 5 7 9; do sbatch --time=01:30:00 --export=ALL,LAYER=$L,MODE=rerun slurm/umt_layer_positional_ablation.sh; done
# One stage: MODE=mass|arms|check|ablate|profiles|summarise, e.g. --export=ALL,LAYER=5,MODE=profiles
# VideoMAE / TimeSformer (6 Nov, same test; VM C1 / Rev1, TF C / Rev; outputs -> {vm,tf}_layer_positional*).
# Selection runs LOCALLY (the DFA parquets stay local), so two GPU jobs per model x layer:
#   1. for M in videomae timesformer; do for L in 5 7 9; do
#        sbatch --time=01:30:00 --export=ALL,MODEL=$M,LAYER=$L,MODE=profiles slurm/umt_layer_positional_ablation.sh; done; done
#   2. rsync profiles_*.npz down, run the select script locally, rsync {vm,tf}_layer_positional/sets_L*.csv up, then
#      MODE=postselect (mass -> ablate -> summarise, spliced then raw-stream) with the same loop (--time=04:00:00).
# Smoke (5 clips, *_smoke dir): --export=ALL,LAYER=5,MAX_CLIPS=5

source $HOME/.tokens   # exports HF_TOKEN
export VIDEO_DIR="/scratch/b6o/tomheslin83.b6o/videos"
export LABELS_PATH="$HOME/labels/labels.json"
export LAYER=${LAYER:-5}
export MODE=${MODE:-chain}
export MAX_CLIPS=${MAX_CLIPS:-0}
export NUM_WORKERS=${NUM_WORKERS:-4}
export SPLICE=${SPLICE:-1}
export MODEL=${MODEL:-umt}
echo "MODEL=$MODEL LAYER=$LAYER MODE=$MODE MAX_CLIPS=$MAX_CLIPS NUM_WORKERS=$NUM_WORKERS SPLICE=$SPLICE"
df -h /dev/shm                                   # loader workers share decoded clips via /dev/shm
SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"
A="python -u src/stage3_analysis/umt_layer_positional_ablation.py"
if [ "$MODE" = select ]; then                 # the dependency job: selection for every layer
    RUN="for L in 5 7 9; do LAYER=\$L python -u src/stage3_analysis/umt_layer_positional_select.py || exit 1; done"
elif [ "$MODE" = rerun ]; then               # new sets on existing profiles + mass: no mass/profiles pass
    RUN="python -u src/stage3_analysis/umt_layer_positional_select.py &&
         MODE=arms $A && MODE=ablate $A && MODE=summarise $A"
elif [ "$MODE" = postselect ]; then          # VM / TF after the local selection: needs sets_L{L}.csv
    # spliced (default) then the raw-stream check (SPLICE=0, own *_L{L} dir + xlayer_summary_raw.csv)
    RUN="for S in 1 0; do SPLICE=\$S MODE=mass $A && SPLICE=\$S MODE=ablate $A && SPLICE=\$S MODE=summarise $A || exit 1; done"
elif [ "$MODE" = chain ]; then
    RUN="MODE=mass $A && MODE=ablate $A && MODE=profiles $A && MODE=summarise $A"
else
    RUN="$A"                                     # the driver reads MODE from the env first
fi

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow scipy \"transformers==5.5.0\" huggingface-hub tqdm \"timm==0.4.12\" easydict &&
        cd \$HOME/temporal-or-textural &&
        unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE &&
        $RUN
    "
