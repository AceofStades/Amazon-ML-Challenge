"""France pairs with the same normalized name and the Source 1 house number present in the target address, but low
address similarity (address IDF cosine < 0.6): are the streets really different, or is normalization failing?"""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
fr = s1.filter(pl.col("country") == "France").select("si")
cols = ["si", "ti", "house_match", "address_idf_cos", "name_exact", "name_idf_cos"]
x = pl.concat([pl.read_parquet(f, columns=cols).join(fr, on="si", how="semi").filter(
    (pl.col("house_match") == 1) & (pl.col("address_idf_cos") < 0.6) & (pl.col("name_idf_cos") >= 0.95)) for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
links = (R("output/final_ctxA/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
x = x.join(links.with_columns(pl.lit(1).alias("acc")), on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
x = x.join(links.select("ti").unique().with_columns(pl.lit(1).alias("held")), on="ti", how="left").with_columns(pl.col("held").fill_null(0))
p1 = pl.concat([pl.read_parquet(f).join(x.select("si", "ti"), on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
x = x.join(p1, on=["si", "ti"], how="left")
print("France same-name, same-house, low address similarity:", x.height, "accepted:", x["acc"].sum(), "free & rejected:", ((x["acc"] == 0) & (x["held"] == 0)).sum())
print(x.with_columns(pl.col("address_idf_cos").cut([0.2, 0.4]).alias("b")).group_by("b").agg(pl.len(), pl.col("acc").mean().round(3), pl.col("p1").median().round(3)).sort("b").rows())
s = x.filter((pl.col("acc") == 0) & (pl.col("held") == 0)).sample(18, seed=4).join(
    s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
    tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
for r in s.iter_rows(named=True):
    print(f"  cos={r['address_idf_cos']:.2f} p1={r['p1']:.2f} | {r['sn'][:24]:24} {r['sa'][:38]:38} || {r['ta'][:40]}")
