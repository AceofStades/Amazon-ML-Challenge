#!/bin/sh
# Final 23:30 file: fleet rebuild (context blend + composition) for US/India with every landed cross-encoder,
# France = file 13's France plus the 0.9-0.98 six-model-blend band (no swaps, brands, nearby added-word pairs) and
# the address-less owner picks (output/final_ctxA_frplus). Keeps ctxA's US/India if the rebuild's CV is lower.
# Usage: ./scripts/final_night.sh NAME [START_HH:MM]
set -e
cd "$(dirname "$0")/.."
NAME=${1:?name}
START=${2:-}
PY=.venv/bin/python
if [ -n "$START" ]; then
    target=$(TZ=Asia/Kolkata date -d "today $START" +%s)
    while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done
fi
FRANCE=output/final_ctxA_frplus/matching_results.tsv
./scripts/final_assemble.sh "$NAME" "$FRANCE" > "logs/final_$NAME.log" 2>&1 || echo "final_assemble failed; falling back to ctxA US/India"
new=$(sed -n "s/^CV $NAME: F0.5 \([0-9.]*\).*/\1/p" "logs/final_$NAME.log")
echo "CV $NAME: ${new:-none} (ctxA 0.99261)"
out=output/final_night
mkdir -p $out
if [ -n "$new" ] && [ -f "output/final_$NAME/matching_results.tsv" ] && $PY -c "import sys; sys.exit(0 if float('$new') >= 0.99255 else 1)"; then
    cp -l "output/final_$NAME/matching_results.tsv" $out/matching_results.tsv 2>/dev/null || cp "output/final_$NAME/matching_results.tsv" $out/
    src="output/final_$NAME"
else
    $PY scripts/merge_france.py --usin output/final_ctxA/matching_results.tsv --france "$FRANCE" --output $out/matching_results.tsv
    src=output/final_ctxA
fi
ln -f "$src/candidate_pairs.tsv" $out/candidate_pairs.tsv
$PY src/utils_validate_submission.py -m $out/matching_results.tsv -c $out/candidate_pairs.tsv | tail -1
echo "US/India from $src; France from $FRANCE"
blend=output/blend_ctxA
[ "$src" = "output/final_$NAME" ] && blend="output/blend_$NAME"
stage="output/to_submit/14_FINAL_context_blend_usin_plus_france_band_and_owner_matching_results.tsv"
rm -f output/to_submit/14_*
ln -f $out/matching_results.tsv "$stage"
cat >> output/to_submit/INDEX.txt <<INDEX

14_FINAL_context_blend_usin_plus_france_band_and_owner_matching_results.tsv  (= $out, built $(TZ=Asia/Kolkata date '+%H:%M'))
   US/India from $src (CV of the rebuild: ${new:-none}; kept only if >= 0.99255, else the 23-model context blend of file 13).
   France: file 13's France + six-model-blend pairs scoring 0.9-0.98 on unclaimed targets (no swaps, brands or nearby
   added-word conflicts) + address-less owner-model picks. File 13 scored 0.989261.
INDEX
$PY -m amzn.package --team-name "Claude's Plan" --submission-dir $out --stage1-dir output/v7 --blend-dir "$blend"
echo "done $out"
