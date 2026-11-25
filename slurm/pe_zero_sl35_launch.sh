#!/bin/bash
# 25/11/26: zeroed-PE real-clip runs on the 696 clips + full characterisation on the SL-35 clips (1,400), all 3 backbones.
# UMT / VM: whole PE zeroed (ZERO_REAL=1). TF: time table and spatial PE zeroed separately (ZERO_REAL=time | space).
# Run from the repo root on Isambard: bash slurm/pe_zero_sl35_launch.sh
set -e
S=slurm; SL="CLIP_CSV=outputs/analysis/sl35_clips.csv"
LIST=$(sbatch --parsable $S/sl35_clip_list.sh)
sb() { sbatch --parsable "$@"; }
# 696 clips: zeroed PE
echo umt zero $(sb --export=ALL,ZERO_REAL=1 $S/umt_layer_pe_characterise.sh)
echo vm zero $(sb --export=ALL,ZERO_REAL=1 $S/vm_layer_pe_characterise.sh)
for z in time space; do echo tf zero$z $(sb --export=ALL,ZERO_REAL=$z $S/tf_layer_pe_characterise.sh); done
# SL-35: normal (blank / R / C / Rev), frozen, zeroed
for m in umt vm tf; do
    D="--dependency=afterok:$LIST"; O="OUT_NAME=${m}_layer_pe_characterise_sl35"
    echo $m sl normal $(sb $D --time=05:00:00 --export=ALL,$SL,$O $S/${m}_layer_pe_characterise.sh)
    echo $m sl frozen $(sb $D --time=03:00:00 --export=ALL,$SL,$O,FROZEN_REAL=1 $S/${m}_layer_pe_characterise.sh)
    Z=$([ $m = tf ] && echo "time space" || echo 1)
    for z in $Z; do echo $m sl zero$z $(sb $D --time=03:00:00 --export=ALL,$SL,$O,ZERO_REAL=$z $S/${m}_layer_pe_characterise.sh); done
done
