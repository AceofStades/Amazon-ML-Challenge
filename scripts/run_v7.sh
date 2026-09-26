#!/bin/sh
# v7: new name normalization (dotted initials, French suffixes), features and stage 1.
set -e
PY=.venv/bin/python
$PY -m amzn.prepare --workers 6
$PY -m amzn.blocking --split train --import-records
$PY -m amzn.blocking --split test --import-records
$PY -m amzn.v3 features --split train --partitions 32
$PY -m amzn.v3 features --split test --partitions 16
$PY -m amzn.train_v4 stage1 p1_train p1_test --model-dir output/v7 --pair-house
