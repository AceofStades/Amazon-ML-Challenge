"""Second context set for the blend (cache/ctx2/{split}.parquet): the entity's rank among all candidate entities of
the target by stage-1 score, a sharper renormalization (p1^2 over the sum of squares), the number of candidates with
p1 >= 0.05, and how many Source 1 entities of the target's country share its normalized name core.
Usage: python scripts/ctx2_features.py train|test"""
import glob
import sys
from pathlib import Path

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
split = sys.argv[1]
pairs = pl.read_parquet(f"cache/ce/{split}_scores.parquet", columns=["si", "ti"]).with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
targets = pairs.select("ti").unique()
p1 = pl.concat([pl.read_parquet(f).join(targets, on="ti", how="semi").filter(pl.col("p1") >= 0.01) for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))])
c = p1.with_columns(pl.col("p1").rank("ordinal", descending=True).over("ti").cast(pl.Float32).alias("rank1"),
                    ((pl.col("p1") ** 2) / (pl.col("p1") ** 2).sum().over("ti")).alias("q1sq"),
                    (pl.col("p1") >= 0.05).sum().over("ti").cast(pl.Float32).alias("m1_05"))
s1 = R(f"cache/{split}_source1.tsv", ["country", "name_core"])
tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["country", "name_core"]) for i in (2, 3)]).with_row_index("ti").with_columns(pl.col("ti").cast(pl.UInt32))
k = s1.group_by("country", "name_core").len().rename({"len": "k_core"})
tk = tg.join(targets, on="ti", how="semi").join(k, on=["country", "name_core"], how="left").select("ti", pl.col("k_core").fill_null(0).cast(pl.Float32))
out = pairs.join(c.select("si", "ti", "rank1", "q1sq", "m1_05"), on=["si", "ti"], how="left").join(tk, on="ti", how="left")
Path("cache/ctx2").mkdir(parents=True, exist_ok=True)
out.write_parquet(f"cache/ctx2/{split}.parquet")
print(split, out.height, out.select(pl.col("rank1").mean(), pl.col("q1sq").mean(), pl.col("m1_05").mean(), pl.col("k_core").mean()).rows())
