#!/bin/sh
# Re-run the fleet blend each time another cross-encoder score set lands (logs/fleet_autoN.log, output/sub_autoN).
cd ~/code/amzn
listing() { ls cache/ce_*/test_scores.parquet 2>/dev/null | sort | tr "\n" " "; }
seen=$(listing)
n=1
while true; do
  now=$(listing)
  if [ "$now" != "$seen" ]; then
    sleep 120
    now=$(listing)
    echo "$(date) run auto$n: $now" >> logs/fleet_auto.log
    ./scripts/fleet_blend.sh auto$n > logs/fleet_auto$n.log 2>&1
    echo "$(date) done auto$n" >> logs/fleet_auto.log
    seen=$now; n=$((n+1))
  fi
  sleep 300
done
