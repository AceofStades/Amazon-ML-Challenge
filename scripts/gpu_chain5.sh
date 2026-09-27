#!/bin/sh
# Fourth local e5-base cross-encoder on hard pairs (seed-191 entities), scored on the shared pair lists.
set -e
PY=.venv/bin/python
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 19 --pairs-file kaggle/data_hard/ce_train_pairs_hard_seed191.parquet --model-dir models/crossencoder_base_hard4
mkdir -p cache/ce_base_hard4
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_hard4 --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_hard4/${s}_scores.parquet; done
