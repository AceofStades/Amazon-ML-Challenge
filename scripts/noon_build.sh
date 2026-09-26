#!/bin/sh
# Wait until the planned noon build, assemble the landed fleet, and stage file 11.
# Run detached from the host, not from a PID-isolated sandbox.
set -eu
cd "$(dirname "$0")/.."
NAME=${1:-fleetA}
WHEN=${2:-2026-09-27 11:00:00}
mkdir -p logs output/to_submit
exec 8>"logs/noon_${NAME}.lock"
flock -n 8 || { echo "noon build already scheduled: $NAME"; exit 0; }
[ ! -f "logs/noon_${NAME}.done" ] || { echo "noon build already finished: $NAME"; exit 0; }
target=$(TZ=Asia/Kolkata date -d "$WHEN" +%s)
echo "$(date -Is) waiting until $WHEN IST for $NAME"
while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done

if [ ! -f "logs/noon_${NAME}.built" ]; then
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
    Submission validator passed. Leaderboard score is unknown; file 10 is still
    unsubmitted as of 2026-09-27 04:00 IST. Confirm composition using its score.
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

The full model list and CV metrics are in logs/noon_${NAME}.log. File 10 was
unsubmitted at handoff, so there was no leaderboard evidence about composition
when this run was scheduled. Review any subsequent file 10 score before upload.
EOF
fi
if ! git diff --quiet -- docs/experiments.md; then
    git add docs/experiments.md
    git commit --only docs/experiments.md -m "Record validated noon fleet submission"
fi
git push origin hmm
touch "logs/noon_${NAME}.done"
echo "$(date -Is) staged $stage"
