"""Holdout macro F0.5 (in candidates) of blend runs, decoded like the hybrid (one winner per target, expected F0.5)."""
import sys

import numpy as np
import polars as pl

from amzn.train_v3 import decode
from amzn.train_v4 import one_to_one

N_HOLD = 22144
for folder in sys.argv[1:]:
    hold = pl.read_parquet(f"{folder}/valid_predictions.parquet")
    scored = one_to_one(hold.select("si", "ti", "p2"), "p2")
    chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p2"].to_numpy(), "expected", 0.05, 0.0)
    picked = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                           "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
        pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.lit(True).alias("c"))
    h = hold.join(picked, on=["si", "ti"], how="left").with_columns(pl.col("c").fill_null(False))
    g = h.group_by("si").agg(pl.col("label").sum().alias("a"), pl.col("c").sum().alias("k"),
                             (pl.col("label").cast(pl.Boolean) & pl.col("c")).sum().alias("tp"))
    a, k, tp = g["a"].to_numpy(), g["k"].to_numpy(), g["tp"].to_numpy()
    f = np.where((a == 0) & (k == 0), 1.0, np.where((a == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * a + k, 1e-9)))
    p = np.clip(hold["p2"].to_numpy(), 1e-6, 1 - 1e-6)
    y = hold["label"].to_numpy()
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    print(f"{folder}: holdout F0.5 {(f.sum() + N_HOLD - len(g)) / N_HOLD:.5f}, logloss (all holdout pairs) {logloss:.5f}", flush=True)
