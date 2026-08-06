#!/bin/bash
#SBATCH --job-name=tot_spliced_vm_k400_l3
#SBATCH --output=spliced_accuracy_vm_k400_l3_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00

# One-off: train_sae_vm_kinetics_l3.sh completed training (best checkpoint saved)
# but timed out during its trailing spliced-accuracy pass. Training itself is not
# resumable at this point — train_sae.py refuses to "resume" a checkpoint already
# at the final epoch (see train_sae.py's resume guard) — so this calls
# spliced_accuracy_vm.py directly against the saved best checkpoint instead of
# resubmitting the full training script.
#
# --eval-clips points at the held-out split train_sae.py persisted during the
# completed training run, so this matches the other layers' methodology exactly
# rather than evaluating on a different clip set.

source $HOME/.tokens

export VIDEO_DIR="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset"
export KINETICS_LABELS_CSV="/scratch/b5bg/tomheslin83.b5bg/data/kinetics400/kinetics-dataset/val.csv"

SIF="$SCRATCHDIR/pytorch_25.05-py3.sif"

apptainer exec --nv \
    --bind $HOME:$HOME \
    --bind $SCRATCHDIR:$SCRATCHDIR \
    $SIF \
    bash -c "
        pip install --quiet av einops pandas pyarrow \"transformers==5.5.0\" huggingface-hub tqdm &&
        cd $HOME/temporal-or-textural &&
        python notebooks/spliced_accuracy_vm.py --layer 3 --dataset-name kinetics400 \
            --sae-checkpoint outputs/sae/sae_vmae_kinetics400_k64_x8_l3_job7ep_best.pt \
            --eval-clips outputs/sae/videomae_kinetics400_held_out_val_clips.json
    "
