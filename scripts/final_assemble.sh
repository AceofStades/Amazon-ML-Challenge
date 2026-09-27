#!/bin/sh
# Final submission assembly: cross-validated fleet blend (every landed cross-encoder), hybrid, composition rescoring
# (with full-competition target context) for
# the labelled countries, France decisions from a finished France file, validation.
# Usage: ./scripts/final_assemble.sh NAME FRANCE_FILE   (outputs output/final_NAME/{matching_results,candidate_pairs}.tsv)
set -e

# Serialize CPU-heavy pipelines with the automatic fleet watcher.
mkdir -p logs
exec 9>logs/blend.lock
flock 9
NAME=${1:?name}
FRANCE=${2:-output/sub_auto2_frsafe/matching_results.tsv}
PY=.venv/bin/python
EXTRAS=${EXTRAS:-name,full,gen,ctx}
EXTRA=""
for dir in cache/ce_*; do
    name=${dir#cache/ce_}
    if [ -f "$dir/train_scores.parquet" ] && [ -f "$dir/test_scores.parquet" ]; then
        EXTRA="$EXTRA --extra-ce $name=$dir"
    fi
done
echo "cross-encoders:$EXTRA"
PYTHONPATH=src $PY scripts/blend_cv.py $EXTRA --extras $EXTRAS --tag $NAME
$PY -m amzn.blend --model-dir output/v7 --extras $EXTRAS $EXTRA --depth 6 --output-dir output/blend_$NAME
$PY -m amzn.hybrid --blend-dir output/blend_$NAME --output-dir output/hybrid_$NAME
mkdir -p output/compose_$NAME output/final_$NAME
$PY scripts/compose.py test --cv-scores output/cv_$NAME.parquet --test-calibrated output/hybrid_$NAME/test_calibrated.parquet \
    --countries US,India --extra --output output/compose_$NAME/matching_results.tsv
$PY scripts/merge_france.py --usin output/compose_$NAME/matching_results.tsv --france $FRANCE \
    --output output/final_$NAME/matching_results.tsv
ln -f output/hybrid_$NAME/candidate_pairs.tsv output/final_$NAME/candidate_pairs.tsv
$PY src/utils_validate_submission.py -m output/final_$NAME/matching_results.tsv -c output/final_$NAME/candidate_pairs.tsv
echo "done output/final_$NAME"
