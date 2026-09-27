#!/bin/sh
# Third local e5-base cross-encoder on hard pairs (seed-161 entities), scored on the shared pair lists.
set -e
PY=.venv/bin/python
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 16 --pairs-file kaggle/data_hard/ce_train_pairs_hard_seed161.parquet --model-dir models/crossencoder_base_hard3
mkdir -p cache/ce_base_hard3
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_hard3 --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_hard3/${s}_scores.parquet; done
