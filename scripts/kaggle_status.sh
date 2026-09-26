#!/bin/sh
# Print the Kaggle status of every training and scoring kernel in kaggle/fleet.tsv, and whether scores are in cache.
cd "$(dirname "$0")/.." || exit 1
K=$HOME/.local/bin/kaggle
grep -v '^#' kaggle/fleet.tsv | while IFS='	' read -r cfg train sdir name note; do
    [ -n "$name" ] || continue
    kg() { if [ "$cfg" = default ]; then "$K" "$@"; else KAGGLE_CONFIG_DIR=$HOME/$cfg "$K" "$@"; fi; }
    score=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['id'])" "kaggle/$sdir/kernel-metadata.json")
    t=$(kg kernels status "$train" 2>&1 | tail -1 | sed 's/.*KernelWorkerStatus\.//; s/"//g')
    s=$(kg kernels status "$score" 2>&1 | tail -1 | sed 's/.*KernelWorkerStatus\.//; s/"//g' | cut -c1-40)
    c=$([ -f "cache/ce_$name/test_scores.parquet" ] && echo "IN CACHE" || echo "-")
    printf '%-10s %-34s train=%-10s score=%-12s %-8s %s\n' "$name" "$train" "$t" "$s" "$c" "$note"
done
