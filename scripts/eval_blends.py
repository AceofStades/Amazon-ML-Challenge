"""Holdout F0.5 (actual) and label-free test estimate for blend runs, same decode as the hybrid."""
import sys

from amzn.estimate import run

for folder in sys.argv[1:]:
    run(folder, "p2", True, folder, "test_scored.parquet", True)
