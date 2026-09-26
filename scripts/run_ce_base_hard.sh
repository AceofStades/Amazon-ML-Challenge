#!/bin/sh
# e5-base cross-encoder on hard pairs (stage-1 candidates with p1 >= 0.01, seed 71), after the GPU is free.
set -e
PY=.venv/bin/python
while pgrep -f "[c]rossenc score" > /dev/null; do sleep 60; done
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 7 --pairs-file kaggle/data_hard/ce_train_pairs_hard_seed71.parquet --model-dir models/crossencoder_base_hard
mkdir -p cache/ce_base_hard
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_hard --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_hard/${s}_scores.parquet; done
