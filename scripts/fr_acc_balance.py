"""Source-balance check of accepted links by house relation x name-change kind, per country. For an entity with 3-4
accepted links whose other links all sit in one source, fill = the link lies in the other source; training truth gives
the true-link reference (0.736 for 2 others, 0.751 for 3), a decoy or wrong owner gives 0.5.
Usage: python scripts/fr_acc_balance.py SUBMISSION"""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
KINDS = ["g_alias", "g_domain", "g_acronym", "g_exact", "g_typo1", "g_drop", "g_add", "g_swap_filler", "g_multi"]
REF = {2: 0.7365, 3: 0.7507}
sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
tg = tg.with_columns(pl.col("entity_id").str.starts_with("S3").cast(pl.Int32).alias("t3"), (pl.col("business_address") == "").alias("noaddr"))
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti", "t3", "noaddr"), left_on=col, right_on="entity_id").select("si", "ti", "t3", "noaddr", "country"))
e = l.group_by("si").agg(pl.len().alias("K"), pl.col("t3").sum().alias("N3"))
x = l.join(e, on="si").filter(pl.col("K").is_in([3, 4])).with_columns((pl.col("K") - 1).alias("kr"), (pl.col("N3") - pl.col("t3")).alias("n3r"))
x = x.filter((pl.col("n3r") == 0) | (pl.col("n3r") == pl.col("kr")))
x = x.with_columns(pl.when(pl.col("n3r") == 0).then(pl.col("t3") == 1).otherwise(pl.col("t3") == 0).alias("fill"),
                   pl.col("kr").replace_strict(REF, return_dtype=pl.Float64).alias("ref"))
f = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos"]).join(x.select("si", "ti"), on=["si", "ti"], how="semi")
               for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
g = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti"] + KINDS).join(x.select("si", "ti"), on=["si", "ti"], how="semi")
x = x.join(f, on=["si", "ti"], how="left").join(g, on=["si", "ti"], how="left")
kind = pl.coalesce([pl.when(pl.col(k) > 0.5).then(pl.lit(k[2:])) for k in KINDS] + [pl.lit("other")])
house = (pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when(pl.col("address_idf_cos").is_null() | (pl.col("address_idf_cos") < 0.6)).then(pl.lit("otherst"))
         .when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")))
x = x.with_columns(kind.alias("kind"), house.alias("house"))
t = x.group_by("country", "house", "kind").agg(pl.len().alias("n"), pl.col("fill").mean().alias("fill"), pl.col("ref").mean().alias("ref"))
t = t.with_columns(((pl.col("fill") - 0.5) / (pl.col("ref") - 0.5)).round(2).alias("pi"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
w = t.filter(pl.col("n") >= 300)
print(w.pivot(on="country", index=["house", "kind"], values="pi").sort("house", "kind"))
print(w.pivot(on="country", index=["house", "kind"], values="n").sort("house", "kind"))
