#!/bin/sh
# After gpu_chain.sh: e5-base boundary specialist (stage-1 pairs with 0.02 <= p1 < 0.98).
set -e
PY=.venv/bin/python
while pgrep -f "^/bin/sh ./scripts/gpu_chain.sh" > /dev/null; do sleep 60; done
$PY -m amzn.crossenc train --base-model intfloat/multilingual-e5-base --seed 20 --pairs-file kaggle/data_hard/ce_train_pairs_boundary_seed201.parquet --model-dir models/crossencoder_base_bnd
mkdir -p cache/ce_base_bnd
for s in train test; do $PY -m amzn.crossenc score --split $s --model-dir models/crossencoder_base_bnd --pairs cache/ce/${s}_scores.parquet --output cache/ce_base_bnd/${s}_scores.parquet; done
