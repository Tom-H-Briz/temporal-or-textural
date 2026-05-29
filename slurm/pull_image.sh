#!/bin/bash
#SBATCH --job-name=pull_pytorch_sif
#SBATCH --output=pull_image_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

singularity pull $SCRATCHDIR/pytorch_25.05-py3.sif docker://nvcr.io/nvidia/pytorch:25.05-py3
