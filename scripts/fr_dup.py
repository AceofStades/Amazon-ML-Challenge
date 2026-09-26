"""France same-house, same-street pairs with an identical name: how stage-1 p1 depends on the name-duplicate count,
compared with US/India test and the training holdout."""
import glob

import polars as pl

ids = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").select("country").with_row_index("si")
cols = ["si", "ti", "house_match", "address_idf_cos", "name_idf_cos", "s_name_dup", "t_name_dup", "s_name_in_targets"]
out = []
for path in sorted(glob.glob("cache/v3/test/part*.parquet")):
    name = path.split("/")[-1]
    x = pl.read_parquet(path, columns=cols).filter((pl.col("house_match") == 1) & (pl.col("address_idf_cos") >= 0.8) & (pl.col("name_idf_cos") >= 0.95))
    x = x.join(pl.read_parquet(f"output/v7/test_p1/{name}"), on=["si", "ti"])
    out.append(x)
x = pl.concat(out).join(ids, on="si")
x = x.with_columns(pl.when(pl.col("s_name_dup") <= 1).then(pl.lit("1")).when(pl.col("s_name_dup") <= 3).then(pl.lit("2-3"))
                   .when(pl.col("s_name_dup") <= 10).then(pl.lit("4-10")).when(pl.col("s_name_dup") <= 50).then(pl.lit("11-50"))
                   .otherwise(pl.lit(">50")).alias("dup"))
t = x.group_by("country", "dup").agg(pl.len().alias("n"), (pl.col("p1") < 0.99).mean().round(4).alias("share_p1_lt_.99"),
                                     (pl.col("p1") < 0.9).mean().round(4).alias("share_lt_.9"), pl.col("p1").mean().round(4).alias("mean_p1"))
pl.Config.set_tbl_rows(40)
print(t.sort("country", "dup"))
