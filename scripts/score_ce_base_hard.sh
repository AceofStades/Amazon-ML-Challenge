#!/bin/sh
# Score the hard-pair e5-base cross-encoder on the shared pairs (resume after a restart).
set -e
mkdir -p cache/ce_base_hard
for s in train test; do .venv/bin/python -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_hard --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_hard/${s}_scores.parquet; done
