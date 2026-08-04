#!/bin/bash
set -euo pipefail

# Pulls position_lock_extraction.py's outputs down from Isambard — outputs/ is
# gitignored so git pull won't bring these; matches your established rsync
# pattern (b5bg.aip2.isambard, ~/temporal-or-textural) from prior syncs.
# Local position_lock_summary.py / scaffold_selection_consolidated.py both
# read from this directory, so this is the only thing you need synced before
# running them locally against the finished SSv2 job.

REMOTE="b5bg.aip2.isambard"
REMOTE_DIR="~/temporal-or-textural/outputs/analysis/position_lock/"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/outputs/analysis/position_lock/"

mkdir -p "$LOCAL_DIR"
rsync -avz "$REMOTE:$REMOTE_DIR" "$LOCAL_DIR"
