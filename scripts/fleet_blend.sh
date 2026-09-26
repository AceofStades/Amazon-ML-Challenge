#!/bin/sh
# Blend every cross-encoder whose train and test scores have landed, check the holdout, then build the hybrid,
# apply the France rules and validate. Usage: ./scripts/fleet_blend.sh NAME  (outputs output/blend_NAME, output/sub_NAME)
set -e
NAME=${1:?name}
PY=.venv/bin/python
EXTRA=""
for pair in b=cache/ce_b c=cache/ce_c L=cache/ce_large base=cache/ce_base basehard=cache/ce_base_hard basehard2=cache/ce_base_hard2 \
            L2=cache/ce_large2 L3=cache/ce_large3 L5=cache/ce_large5 L10=cache/ce_large10 \
            bge=cache/ce_bge bge6=cache/ce_bge6 bge11=cache/ce_bge11 xlmr7=cache/ce_xlmr7 xlmr12=cache/ce_xlmr12 \
            e5i8=cache/ce_e5i8 e5i13=cache/ce_e5i13 bgem3=cache/ce_bgem3 \
            labse14=cache/ce_labse14 bgem3_15=cache/ce_bgem3_15 L16=cache/ce_large16 bge17=cache/ce_bge17; do
    dir=${pair#*=}
    if [ -f "$dir/train_scores.parquet" ] && [ -f "$dir/test_scores.parquet" ]; then
        EXTRA="$EXTRA --extra-ce $pair"
    fi
done
echo "cross-encoders:$EXTRA"
$PY -m amzn.blend --model-dir output/v7 --extras name,full,gen $EXTRA --depth 6 --output-dir output/blend_$NAME
PYTHONPATH=src $PY scripts/eval_holdout.py output/blend_v7_genLB output/blend_$NAME
$PY -m amzn.hybrid --blend-dir output/blend_$NAME --output-dir output/hybrid_$NAME
mkdir -p output/sub_$NAME
# Raw (uncalibrated) decisions for countries without labels: the pool the France rule restores patterns from.
RAW=output/hybrid_L_frnone/matching_results.tsv
if [ ! -f "$RAW" ]; then
    $PY -m amzn.hybrid --blend-dir output/blend_$NAME --fallback-calibration none --output-dir output/hybrid_${NAME}_raw
    RAW=output/hybrid_${NAME}_raw/matching_results.tsv
fi
$PY -m amzn.france_rule --calibrated output/hybrid_$NAME/matching_results.tsv --raw $RAW \
    --drop-conflicts --drop-swaps --add-initials --output output/sub_$NAME/matching_results.tsv
ln -f output/hybrid_$NAME/candidate_pairs.tsv output/sub_$NAME/candidate_pairs.tsv
$PY src/utils_validate_submission.py -m output/sub_$NAME/matching_results.tsv -c output/sub_$NAME/candidate_pairs.tsv | tail -1
echo "done output/sub_$NAME"
