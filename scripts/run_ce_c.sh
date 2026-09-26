#!/bin/sh
# Third cross-encoder (seed 3): train, then score the same pairs as the first two.
set -e
PY=.venv/bin/python
$PY -m amzn.crossenc train --seed 3 --entities 600000 --model-dir models/crossencoder_c
mkdir -p cache/ce_c
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_c --pairs cache/ce/${s}_scores.parquet --output cache/ce_c/${s}_scores.parquet; done
