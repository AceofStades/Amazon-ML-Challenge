#!/bin/sh
# Wait for the full-train feature build, then run v4.
while pgrep -f "amzn[.]v3 features --split train" > /dev/null; do sleep 20; done
exec ./scripts/run_v4.sh
