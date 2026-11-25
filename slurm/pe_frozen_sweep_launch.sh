#!/bin/bash
# 25/11/26: frozen-at-k sweep on the 696 real clips (R only): every frame / tubelet given position k's PE, k = 1..n-1
# (k = 0 is the existing frozen run). -> outputs/analysis/{umt,vm,tf}_layer_pe_characterise/frozen{k}_L{5,7,9}.npz
# Run from the repo root on Isambard: bash slurm/pe_frozen_sweep_launch.sh
for k in $(seq 1 11); do echo umt k$k $(sbatch --parsable --export=ALL,FROZEN_REAL=1,FROZEN_K=$k slurm/umt_layer_pe_characterise.sh); done
for m in vm tf; do
    for k in $(seq 1 7); do echo $m k$k $(sbatch --parsable --export=ALL,FROZEN_REAL=1,FROZEN_K=$k slurm/${m}_layer_pe_characterise.sh); done
done
