"""Source 1 entities that share their normalized name with another entity at the same normalized address (or the same
street and house): their targets are indistinguishable, so one-winner-per-target splits them. Test by country, train."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
for split in ("train", "test"):
    s = R(f"cache/{split}_source1.tsv", ["country", "name_core", "address_core", "house"])
    s = s.with_columns(pl.len().over("country", "name_core", "address_core").alias("dup_addr"),
                       pl.len().over("country", "name_core").alias("dup_name"))
    t = s.group_by("country").agg(pl.len().alias("E"), (pl.col("dup_addr") > 1).mean().round(4).alias("share_same_name_addr"),
                                  (pl.col("dup_name") > 1).mean().round(4).alias("share_same_name"),
                                  pl.col("dup_name").mean().round(2).alias("mean_name_group"))
    print(split, t.sort("country"))
