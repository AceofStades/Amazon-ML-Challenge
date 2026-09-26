#!/bin/sh
# Cross-encoder on multilingual-e5-base (MIT, 278M): train on seed-4 entities, score the shared pairs.
set -e
PY=.venv/bin/python
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 4 --entities 600000 --model-dir models/crossencoder_base
mkdir -p cache/ce_base
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base --pairs cache/ce/${s}_scores.parquet --output cache/ce_base/${s}_scores.parquet; done
