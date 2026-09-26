#!/bin/sh
# Sequential GPU chain (no overlap): hard e5-base test scoring, then the second hard e5-base (seed 141).
set -e
PY=.venv/bin/python
$PY -m amzn.crossenc score --split test --model-dir models/crossencoder_base_hard --pairs cache/ce/test_scores.parquet --output cache/ce_base_hard/test_scores.parquet
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 14 --pairs-file kaggle/data_hard/ce_train_pairs_hard_seed141.parquet --model-dir models/crossencoder_base_hard2
mkdir -p cache/ce_base_hard2
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_hard2 --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_hard2/${s}_scores.parquet; done
