#!/bin/sh
# Wait for the boundary chain, then recover hard2 and boundary scores without GPU overlap.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
mkdir -p logs cache/ce_base_hard2
exec 9>logs/gpu_recover_hard2.lock
if ! flock -n 9; then
    printf '%s\n' 'hard2 recovery watcher already running'
    exit 0
fi
boundary_pid=${1:-}
case "$boundary_pid" in
    ''|*[!0-9]*)
        printf '%s\n' 'usage: gpu_recover_hard2.sh BOUNDARY_CHAIN_PID' >&2
        exit 2
        ;;
esac
printf '%s\n' "waiting for boundary chain PID $boundary_pid and GPU scoring workers"
while :; do
    if kill -0 "$boundary_pid" 2>/dev/null ||
       pgrep -f '(^|/)(sh|bash) ([^ ]*/)?scripts/gpu_chain2[.]sh([ ]|$)' >/dev/null ||
       pgrep -f '(^|/)python([0-9.]+)? -m amzn[.]crossenc (train|score)([ ]|$)' >/dev/null; then
        sleep 30
        continue
    fi
    gpu_used=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits)
    case "$gpu_used" in
        ''|*[!0-9]*)
            printf '%s\n' "cannot verify idle GPU: $gpu_used" >&2
            exit 2
            ;;
    esac
    if [ "$gpu_used" -ge 1024 ]; then
        sleep 30
        continue
    fi
    break
done
printf '%s\n' 'boundary chain finished; starting checkpointed local score recovery'
status=0
for cache_name in base_hard2 base_bnd; do
    model="models/crossencoder_${cache_name}"
    if [ ! -f "$model/head.pt" ] || [ ! -f "$model/config.json" ] ||
       { [ ! -f "$model/model.safetensors" ] && [ ! -f "$model/model.safetensors.index.json" ]; }; then
        printf '%s\n' "$cache_name has no final model checkpoint; recovery cannot score it" >&2
        status=1
        continue
    fi
    mkdir -p "cache/ce_${cache_name}"
    for split in train test; do
        if .venv/bin/python -c 'import sys, pyarrow.parquet as pq; name, split=sys.argv[1:]; expected=pq.read_metadata(f"cache/ce/{split}_scores.parquet"); actual=pq.read_metadata(f"cache/ce_{name}/{split}_scores.parquet"); assert actual.num_rows == expected.num_rows and {"si", "ti", "ce_logit"}.issubset(actual.schema.names)' "$cache_name" "$split" 2>/dev/null; then
            printf '%s\n' "$cache_name $split scores already complete; preserving existing file"
            continue
        fi
        if .venv/bin/python scripts/score_ce_checkpointed.py \
            --split "$split" --model-dir "$model" \
            --pairs "cache/ce/${split}_scores.parquet" \
            --output "cache/ce_${cache_name}/${split}_scores.parquet" \
            --batch-size 128 --chunk-rows 250000; then
            continue
        fi
        printf '%s\n' "$cache_name $split recovery failed; completed chunks remain resumable" >&2
        status=1
    done
done
printf '%s\n' "local score recovery finished with status $status"
exit "$status"
