#!/bin/sh
# Late fleet rebuild: waits for the Kaggle scoring kernels 31-34 to land (or a cutoff), then runs the context-blend
# pipeline with every landed cross-encoder and file 11's France, and the composition CV for the test-like view.
# Nothing is staged: compare the printed CVs with ctxA (blend CV 0.99261, test-like 0.99314; composition 0.99327).
# Usage: ./scripts/fleet_final.sh NAME CUTOFF_HH:MM
cd "$(dirname "$0")/.."
NAME=${1:?name}
CUTOFF=$(TZ=Asia/Kolkata date -d "today ${2:?cutoff}" +%s)
PY=.venv/bin/python
while [ "$(date +%s)" -lt "$CUTOFF" ]; do
    n=0
    for m in xlmr31 bge32 e5i33 bgem3_34; do
        [ -f cache/ce_$m/train_scores.parquet ] && [ -f cache/ce_$m/test_scores.parquet ] && n=$((n + 1))
    done
    [ "$n" -eq 4 ] && break
    sleep 120
done
echo "start $(TZ=Asia/Kolkata date +%H:%M): $(ls -d cache/ce_* | wc -l) cross-encoder dirs"
./scripts/final_assemble.sh "$NAME" output/final_pre1_frcu/matching_results.tsv > "logs/final_$NAME.log" 2>&1
echo "assemble exit $? at $(TZ=Asia/Kolkata date +%H:%M)"
grep "^CV $NAME" "logs/final_$NAME.log"
PYTHONPATH=src $PY scripts/compose.py cv --scores output/cv_$NAME.parquet --extra > "logs/compose_cv_${NAME}_extra.log" 2>&1
tail -n 3 "logs/compose_cv_${NAME}_extra.log"
PYTHONPATH=src $PY scripts/cv_error_audit.py output/cv_$NAME.parquet 2>&1 | grep "^macro"
echo "done $(TZ=Asia/Kolkata date +%H:%M)"
