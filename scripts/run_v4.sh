#!/bin/sh
# Full v4 pipeline after encoder kNN tables exist for both splits.
set -e
.venv/bin/python -m amzn.v3 features --split test --partitions 16
.venv/bin/python -m amzn.train_v4 stage1 p1_train p1_test stage2 p2_train p2_test evaluate write --output-dir output/v4
