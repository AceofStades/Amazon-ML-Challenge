#!/bin/sh
# After gpu_chain2.sh: redo the hard e5-base (seed 141) test scoring, which crashed with "free(): invalid pointer"
# at 4.9M of 8.8M pairs in gpu_chain.sh. Retries once.
PY=.venv/bin/python
while pgrep -f "^/bin/sh ./scripts/gpu_chain2.sh" > /dev/null; do sleep 60; done
for attempt in 1 2; do
    $PY -m amzn.crossenc score --split test --model-dir models/crossencoder_base_hard2 --pairs cache/ce/test_scores.parquet \
        --output cache/ce_base_hard2/test_scores.parquet && break
done
