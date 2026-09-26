#!/bin/sh
# Wait until the planned noon build, assemble the landed fleet, and stage file 11.
# Run detached from the host, not from a PID-isolated sandbox.
set -eu
cd "$(dirname "$0")/.."
NAME=${1:-fleetA}
WHEN=${2:-2026-09-27 11:00:00}
UNTIL=${3:-2026-09-27 12:00:00}
mkdir -p logs output/to_submit
exec 8>"logs/noon_${NAME}.lock"
flock -n 8 || { echo "noon build already scheduled: $NAME"; exit 0; }
[ ! -f "logs/noon_${NAME}.done" ] || { echo "noon build already finished: $NAME"; exit 0; }
target=$(TZ=Asia/Kolkata date -d "$WHEN" +%s)
cutoff=$(TZ=Asia/Kolkata date -d "$UNTIL" +%s)
[ "$cutoff" -ge "$target" ] || { echo "readiness cutoff precedes start time" >&2; exit 2; }
echo "$(date -Is) waiting until $WHEN IST for $NAME"
while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done

if [ ! -f "logs/noon_${NAME}.built" ]; then
    while :; do
        ready=0
        .venv/bin/python scripts/fleet_ready.py --local base_hard2 --local base_bnd || ready=$?
        case "$ready" in
            0) break ;;
            1) ;;
            *) echo "fleet readiness check failed" >&2; exit "$ready" ;;
        esac
        if [ "$(date +%s)" -ge "$cutoff" ]; then
            echo "Readiness cutoff reached; building landed scores. Review pending models before submission."
            break
        fi
        sleep 60
    done
    ./scripts/final_assemble.sh "$NAME"
    touch "logs/noon_${NAME}.built"
fi
stage="output/to_submit/11_${NAME}_blend_composition_france_matching_results.tsv"
if [ -e "$stage" ]; then
    [ "$stage" -ef "output/final_${NAME}/matching_results.tsv" ] || {
        echo "staging path already contains a different file: $stage" >&2
        exit 1
    }
else
    ln "output/final_${NAME}/matching_results.tsv" "$stage"
fi
if ! grep -Fq "$stage" output/to_submit/INDEX.txt 2>/dev/null; then
cat >> output/to_submit/INDEX.txt <<EOF

11. $stage
    US/India: every cross-encoder with both score files at build time, 10-fold CV,
    segment calibration and composition rescoring. France: sub_auto2_frsafe.
    Submission validator passed. Leaderboard score is unknown. File 10 is held
    while the fleet finishes. Composition has CV evidence, no isolated leaderboard test.
    Build metrics and selected models: logs/noon_${NAME}.log.
EOF
fi
if ! grep -Fq "## Noon fleet build: $NAME (" docs/experiments.md; then
cat >> docs/experiments.md <<EOF

## Noon fleet build: $NAME ($(TZ=Asia/Kolkata date '+%F %H:%M') IST)

The scheduled assembly completed with every cross-encoder that had both score
files when the build began. It uses 10-fold CV, the full blend, segment calibration,
US/India composition rescoring, and the existing safe France decisions. The
submission validator passed. File 11 is staged at $stage as a hardlink; its
candidate file is also hardlinked. The leaderboard score remains unknown.

The full model list and CV metrics are in logs/noon_${NAME}.log. File 10 is held
while the fleet finishes. Its blend, composition, and France changes are confounded;
use otherwise identical fleet variants if a leaderboard test of composition is needed.
EOF
fi
if ! git diff HEAD --quiet -- docs/experiments.md; then
    git add docs/experiments.md
    git commit --only docs/experiments.md -m "Record validated noon fleet submission"
fi
git push origin hmm
touch "logs/noon_${NAME}.done"
echo "$(date -Is) staged $stage"
