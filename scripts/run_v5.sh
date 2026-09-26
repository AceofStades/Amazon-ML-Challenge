#!/bin/sh
# v5: stage 2 with target-pair cluster features and cross-encoder logits on top of v4 stage 1.
set -e
PY=.venv/bin/python
$PY -m amzn.tt build --split train
$PY -m amzn.tt build --split test
$PY -m amzn.tt train
$PY -m amzn.tt aggregate --split train
$PY -m amzn.tt aggregate --split test
mkdir -p cache/ce
$PY -m amzn.crossenc select --split train --output cache/ce/train_pairs.parquet
$PY -m amzn.crossenc select --split test --output cache/ce/test_pairs.parquet
$PY -m amzn.crossenc score --split train --pairs cache/ce/train_pairs.parquet --output cache/ce/train_scores.parquet
$PY -m amzn.crossenc score --split test --pairs cache/ce/test_pairs.parquet --output cache/ce/test_scores.parquet
$PY -m amzn.train_v4 stage2 p2_train p2_test evaluate write --model-dir output/v4 --stage2-dir output/v5 --output-dir output/v5 --calibrate --extra "cache/tt/{split}_aggregates.parquet" --extra "cache/ce/{split}_scores.parquet"
