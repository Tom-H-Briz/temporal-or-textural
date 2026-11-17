#!/bin/bash
#SBATCH --job-name=tot_move_analysis
#SBATCH --output=/scratch/b6o/tomheslin83.b6o/move_analysis_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

# One-off, CPU-only (17/11/26): move outputs/analysis from home (101G quota, full) to scratch,
# then symlink it back so scripts keep writing to ROOT/outputs/analysis. Rerunnable.
SRC=$HOME/temporal-or-textural/outputs/analysis
DST=$SCRATCHDIR/temporal-or-textural/outputs/analysis
[ -L "$SRC" ] && { echo "already a symlink -> $(readlink $SRC)"; exit 0; }
mkdir -p "$DST"
rsync -a --remove-source-files "$SRC/" "$DST/" || { echo "rsync failed, rerun"; exit 1; }
LEFT=$(find "$SRC" -type f | wc -l)
[ "$LEFT" -ne 0 ] && { echo "$LEFT files still in home, not linking"; exit 1; }
find "$SRC" -type d -empty -delete
ln -s "$DST" "$SRC" && echo "linked: $(ls "$SRC/" | wc -l) entries" && quota -s
