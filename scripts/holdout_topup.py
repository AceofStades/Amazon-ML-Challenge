"""Holdout: for entities with k accepted pairs, precision of the best rejected candidate at the Source 1 house and street,
by k and by its score; and the macro F0.5 change from adding it. Decisions from column argv[2] (p1 or p2)."""
import glob
import sys

import numpy as np
import polars as pl

from amzn.train_v3 import decode
from amzn.train_v4 import one_to_one

folder, col = sys.argv[1], sys.argv[2]
hold = pl.read_parquet(f"{folder}/valid_predictions.parquet")
N_HOLD = 22144
scored = one_to_one(hold.select("si", "ti", col), col)
chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored[col].to_numpy(), "expected", 0.05, 0.0)
acc = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                    "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
    pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.lit(1).alias("acc"))
truth = hold.filter(pl.col("label") == 1).select("si", "ti")
feat = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos"]).join(hold.select("si").unique(), on="si", how="semi")
                  for p in sorted(glob.glob("cache/v3/train/part*.parquet"))])
x = hold.join(acc, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0)).join(feat, on=["si", "ti"], how="left")
k = x.group_by("si").agg(pl.col("acc").sum().alias("k"))
taken = acc.select("ti")
rej = (x.filter((pl.col("acc") == 0) & (pl.col("house_match") == 1) & (pl.col("address_idf_cos") >= 0.8))
       .join(taken, on="ti", how="anti").sort(col, descending=True).unique("si", keep="first").join(k, on="si"))


def macro(accepted):
    a = truth.group_by("si").len().rename({"len": "a"})
    kk = accepted.join(hold.select("si", "ti", "label"), on=["si", "ti"], how="left").group_by("si").agg(
        pl.len().alias("k"), pl.col("label").sum().alias("tp"))
    g = a.join(kk, on="si", how="full", coalesce=True).fill_null(0)
    f = np.where((g["a"] == 0) & (g["k"] == 0), 1.0, np.where((g["a"] == 0) | (g["k"] == 0), 0.0,
                 1.25 * g["tp"].to_numpy() / np.maximum(0.25 * g["a"].to_numpy() + g["k"].to_numpy(), 1e-9)))
    return (f.sum() + N_HOLD - g.height) / N_HOLD


base = macro(acc.select("si", "ti"))
print(f"{col} decisions: holdout F0.5 {base:.5f}")
for kk in (0, 1, 2, 3):
    for lo in (0.05, 0.2, 0.5):
        add = rej.filter((pl.col("k") == kk) & (pl.col(col) >= lo))
        if not add.height:
            continue
        new = macro(pl.concat([acc.select("si", "ti"), add.select("si", "ti")]))
        print(f"  k={kk} best rejected same-address candidate with {col}>={lo}: {add.height:,} pairs, "
              f"precision {add['label'].mean():.3f}, F0.5 {new - base:+.5f}")
