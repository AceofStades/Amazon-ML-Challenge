#!/bin/sh
# Score the second cross-encoder on the same pairs as the first once it is trained.
set -e
while [ ! -f models/crossencoder_b/head.pt ]; do sleep 30; done
mkdir -p cache/ce_b
for s in train test; do .venv/bin/python -m amzn.crossenc score --split $s --model-dir models/crossencoder_b --pairs cache/ce/${s}_scores.parquet --output cache/ce_b/${s}_scores.parquet; done
