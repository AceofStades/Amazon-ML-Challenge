#!/bin/sh
# Wait for cross-encoder test scores, then run v5 stage 2 (extras) on top of v4 stage 1.
while [ ! -f cache/ce/test_scores.parquet ]; do sleep 20; done
exec .venv/bin/python -m amzn.train_v4 stage2 p2_train p2_test evaluate write --model-dir output/v4 --stage2-dir output/v5 --output-dir output/v5 --calibrate --extra "cache/tt/{split}_aggregates.parquet" --extra "cache/ce/{split}_scores.parquet"
