#!/bin/sh
# Second hard-pair e5-base cross-encoder (seed 141), after the GPU is free.
set -e
PY=.venv/bin/python
while pgrep -f "^.venv/bin/python -m amzn.crossenc" > /dev/null; do sleep 60; done
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 14 --pairs-file kaggle/data_hard/ce_train_pairs_hard_seed141.parquet --model-dir models/crossencoder_base_hard2
mkdir -p cache/ce_base_hard2
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_hard2 --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_hard2/${s}_scores.parquet; done
