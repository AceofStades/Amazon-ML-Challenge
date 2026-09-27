"""Entity-side context for the blend (cache/ectx/{split}.parquet): over all candidates of the Source 1 entity
(stage-1 p1 >= 0.01): the pair's p1 over the entity's sum, the entity's number of candidates with p1 >= 0.5, this
target's rank among them, and the entity's best other p1. Usage: python scripts/ectx_features.py train|test"""
import glob
import sys
from pathlib import Path

import polars as pl

split = sys.argv[1]
pairs = pl.read_parquet(f"cache/ce/{split}_scores.parquet", columns=["si", "ti"]).with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
ents = pairs.select("si").unique()
p1 = pl.concat([pl.read_parquet(f).join(ents, on="si", how="semi").filter(pl.col("p1") >= 0.01) for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))])
c = p1.with_columns((pl.col("p1") / pl.col("p1").sum().over("si")).alias("e_q1"),
                    (pl.col("p1") >= 0.5).sum().over("si").cast(pl.Float32).alias("e_m05"),
                    pl.col("p1").rank("ordinal", descending=True).over("si").cast(pl.Float32).alias("e_rank1"))
best2 = c.group_by("si").agg(pl.col("p1").sort(descending=True).head(2).alias("top"))
c = c.join(best2, on="si").with_columns(
    pl.when(pl.col("p1") >= pl.col("top").list.get(0)).then(pl.col("top").list.get(1, null_on_oob=True)).otherwise(pl.col("top").list.get(0))
    .fill_null(0.0).alias("e_top_other"))
out = pairs.join(c.select("si", "ti", "e_q1", "e_m05", "e_rank1", "e_top_other"), on=["si", "ti"], how="left")
Path("cache/ectx").mkdir(parents=True, exist_ok=True)
out.write_parquet(f"cache/ectx/{split}.parquet")
print(split, out.height, out.select(pl.col("e_q1").mean(), pl.col("e_m05").mean(), pl.col("e_rank1").mean()).rows())
