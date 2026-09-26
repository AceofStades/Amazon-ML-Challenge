#!/bin/sh
# Once e5-base test scores exist: blend with it (with and without gen features) and evaluate on the holdout.
set -e
PY=.venv/bin/python
while [ ! -f cache/ce_base/test_scores.parquet ]; do sleep 60; done
sleep 30
$PY -m amzn.blend --model-dir output/v7 --extras name,full --extra-ce b=cache/ce_b --extra-ce c=cache/ce_c --extra-ce L=cache/ce_large --extra-ce base=cache/ce_base --depth 6 --output-dir output/blend_v7_fullLB
$PY -m amzn.blend --model-dir output/v7 --extras name,full,gen --extra-ce b=cache/ce_b --extra-ce c=cache/ce_c --extra-ce L=cache/ce_large --extra-ce base=cache/ce_base --depth 6 --output-dir output/blend_v7_genLB
PYTHONPATH=src $PY scripts/eval_blends.py output/blend_v7_fullLB output/blend_v7_genLB
