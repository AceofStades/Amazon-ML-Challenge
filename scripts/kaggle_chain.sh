#!/bin/sh
# One Kaggle cross-encoder chain: wait for the training kernel, push its scoring kernel, wait, download the scores
# into cache/ce_NAME (train_scores.parquet, then test_scores.parquet, which scripts/fleet_auto.sh picks up).
# Usage: scripts/kaggle_chain.sh CONFIG_DIR TRAIN_KERNEL SCORE_KERNEL_DIR NAME
#   CONFIG_DIR: KAGGLE_CONFIG_DIR of the owning account, or "default" for ~/.kaggle
# Safe to re-run: exits if the scores are already in cache, and does not push the scoring kernel again when it is
# already queued, running or complete (a second push would start a new version and burn GPU quota).
cd "$(dirname "$0")/.." || exit 1
CFG=$1; TRAIN=$2; SDIR=$3; NAME=$4
K=$HOME/.local/bin/kaggle
kg() { if [ "$CFG" = default ]; then "$K" "$@"; else KAGGLE_CONFIG_DIR=$CFG "$K" "$@"; fi; }
# Kernel status line; transient API failures (no status in the reply) are retried so they are never read as a result.
status() {
    for i in 1 2 3 4 5; do
        s=$(kg kernels status "$1" 2>&1 | tail -1)
        echo "$s" | grep -q "KernelWorkerStatus\|Cannot access kernel" && break
        sleep 60
    done
    echo "$s"
}
wait_done() { while :; do s=$(status "$1"); echo "$s" | grep -q "KernelWorkerStatus\.\(COMPLETE\|ERROR\|CANCEL\)" && { echo "$s"; return; }; sleep 180; done; }
log() { echo "$(date '+%F %T') $NAME: $*"; }
SCORE=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['id'])" "$SDIR/kernel-metadata.json")
OUT=cache/ce_$NAME
[ -f "$OUT/test_scores.parquet" ] && { log "scores already in $OUT"; exit 0; }
s=$(status "$SCORE")
if echo "$s" | grep -q "KernelWorkerStatus\.\(QUEUED\|RUNNING\|COMPLETE\)"; then
    log "scoring kernel already pushed: $s"
else
    st=$(wait_done "$TRAIN"); log "train: $st"
    echo "$st" | grep -q "KernelWorkerStatus\.COMPLETE" || { log "training did not complete; stopping"; exit 1; }
    # Pushes fail while the account already runs two GPU sessions; retry for up to two hours.
    for i in $(seq 1 24); do
        r=$(kg kernels push -p "$SDIR" --accelerator NvidiaTeslaT4 2>&1 | tail -1); log "push: $r"
        echo "$r" | grep -q -i successfully && break
        sleep 300
    done
    echo "$r" | grep -q -i successfully || { log "scoring kernel could not be pushed; stopping"; exit 1; }
    sleep 120
fi
st=$(wait_done "$SCORE"); log "score: $st"
echo "$st" | grep -q "KernelWorkerStatus\.COMPLETE" || { log "scoring did not complete; stopping"; exit 1; }
tmp=cache/kaggle_out/$NAME; mkdir -p "$tmp"
for i in 1 2 3; do kg kernels output "$SCORE" -p "$tmp" -o 2>&1 | tail -1; [ -f "$tmp/scores_test.parquet" ] && break; sleep 60; done
[ -f "$tmp/scores_test.parquet" ] || { log "download failed"; exit 1; }
mkdir -p "$OUT" && mv "$tmp/scores_train.parquet" "$OUT/train_scores.parquet" && mv "$tmp/scores_test.parquet" "$OUT/test_scores.parquet"
rm -rf "$tmp"; log "scores in $OUT"
