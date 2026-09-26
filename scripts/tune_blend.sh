#!/bin/sh
# Blend tuning on the genLB feature set, holdout only.
PY=.venv/bin/python
B="--model-dir output/v7 --extras name,full,gen --extra-ce b=cache/ce_b --extra-ce c=cache/ce_c --extra-ce L=cache/ce_large --extra-ce base=cache/ce_base --holdout-only"
$PY -m amzn.blend $B --depth 6 --eta 0.05 --output-dir output/tune_d6e05
$PY -m amzn.blend $B --depth 8 --eta 0.05 --min-child-weight 3 --output-dir output/tune_d8e05
$PY -m amzn.blend $B --depth 6 --eta 0.05 --bag 5 --subsample 0.8 --colsample 0.8 --output-dir output/tune_d6bag5
PYTHONPATH=src $PY scripts/eval_holdout.py output/blend_v7_genLB output/tune_d6e05 output/tune_d8e05 output/tune_d6bag5
