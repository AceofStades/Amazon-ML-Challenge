#!/bin/sh
# Start (or resume) every chain listed in kaggle/fleet.tsv as a detached background process on this machine.
# Chains already running are left alone. Logs: logs/kaggle/NAME.log. Status of all kernels: scripts/kaggle_status.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs/kaggle
grep -v '^#' kaggle/fleet.tsv | while IFS='	' read -r cfg train sdir name note; do
    [ -n "$name" ] || continue
    [ "$cfg" = default ] || cfg=$HOME/$cfg
    if pgrep -f "kaggle_chain.sh .* $name\$" > /dev/null; then echo "running: $name"; continue; fi
    if [ -f "cache/ce_$name/test_scores.parquet" ]; then echo "done: $name"; continue; fi
    setsid nohup ./scripts/kaggle_chain.sh "$cfg" "$train" "kaggle/$sdir" "$name" >> "logs/kaggle/$name.log" 2>&1 < /dev/null &
    echo "started: $name"
done
