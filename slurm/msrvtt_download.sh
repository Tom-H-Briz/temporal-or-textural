#!/bin/bash
#SBATCH --job-name=tot_msrvtt_dl
#SBATCH --output=/scratch/b6o/tomheslin83.b6o/msrvtt_download_%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

# One-off, CPU-only (24/11/26): MSRVTT videos for the UMT 25M zero-shot gate
# (umt25m_brief_241126.md §1). Source = DATASET.md link. Rerunnable: skips finished steps.
URL="https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip"
DST=$SCRATCHDIR/data/msrvtt
mkdir -p "$DST" && cd "$DST"
[ -f MSRVTT.zip.done ] || { wget -c -q "$URL" -O MSRVTT.zip && touch MSRVTT.zip.done; } \
    || { echo "download failed, rerun (wget -c resumes)"; exit 1; }
[ -f unzip.done ] || { unzip -q -o MSRVTT.zip && touch unzip.done; } || { echo "unzip failed"; exit 1; }
echo "mp4 count: $(find "$DST" -name '*.mp4' | wc -l)  (MSRVTT has 10,000)"
echo "annotation files:"; find "$DST" -maxdepth 4 \( -name '*.json' -o -name '*.txt' -o -name '*.csv' -o -name '*.pkl' \) | head -20
du -sh "$DST"
