"""Entities a submission leaves empty, by country: what their best candidates look like (stage-1 p1, house relation,
name difference) and whether another entity holds those targets. A high share of empty entities whose same-house,
same-name target went to another entity would point at false merges between same-name entities.
Usage: python scripts/empty_entities.py SUBMISSION"""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
m = R(sub).join(s1, left_on="source1_entity_id", right_on="entity_id")
empty = m.filter(pl.col("matched_entity_ids") == "").select("si", "country")
col = "matched_entity_ids"
l = (m.filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(tg, left_on=col, right_on="entity_id").select("si", "ti"))
held = l.select("ti").unique().with_columns(pl.lit(1).alias("held"))
p1 = pl.concat([pl.read_parquet(f).join(empty, on="si", how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
feats = pl.concat([pl.read_parquet(f, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"])
                   .join(empty, on="si", how="semi") for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
x = p1.join(feats, on=["si", "ti"], how="left").join(held, on="ti", how="left").with_columns(pl.col("held").fill_null(0))
x = x.join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti", how="left")
best = x.sort("p1", descending=True).unique("si", keep="first")
best = empty.join(best, on="si", how="left").with_columns(
    pl.when(pl.col("p1").is_null()).then(pl.lit("none"))
    .when(pl.col("p1") < 0.1).then(pl.lit("p<0.1"))
    .when(pl.col("p1") < 0.5).then(pl.lit("p0.1-0.5"))
    .when(pl.col("p1") < 0.9).then(pl.lit("p0.5-0.9")).otherwise(pl.lit("p>=0.9")).alias("pb"),
    pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when(pl.col("house_match") == 1).then(pl.lit("same"))
    .when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")).alias("h"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
n = s1.group_by("country").len().rename({"len": "E"})
print(empty.group_by("country").len().join(n, on="country").with_columns((pl.col("len") / pl.col("E")).alias("empty_share")))
t = best.group_by("country", "pb", "h", "held").len().join(n, on="country").with_columns((pl.col("len") / pl.col("E") * 1000).round(3).alias("per_1000E"))
print(t.filter(pl.col("pb").is_in(["p0.5-0.9", "p>=0.9"])).sort("country", "pb", "h", "held"))
print(best.group_by("country", "pb").len().join(n, on="country").with_columns((pl.col("len") / pl.col("E") * 1000).round(3).alias("per_1000E")).sort("country", "pb"))
best.write_parquet("cache/empty_best.parquet")
