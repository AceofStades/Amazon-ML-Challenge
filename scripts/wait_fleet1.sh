#!/bin/sh
# Run the first fleet blend once the hard e5-base and L2 scores exist.
while [ ! -f cache/ce_base_hard/test_scores.parquet ] || [ ! -f cache/ce_large2/test_scores.parquet ]; do sleep 120; done
sleep 60
./scripts/fleet_blend.sh fleet1
