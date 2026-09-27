"""US/India: pairs the 22-model blend scores above 0.98 that the submission does not accept, split by whether the
target is free or held by another entity, and by house relation; plus the (0.9, 0.98] band."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
ui = s1.filter(pl.col("country") != "France").select("si", "country")
sc = pl.read_parquet("output/blend_fleetA/test_scored.parquet", columns=["si", "ti", "p2"]).join(ui, on="si")
links = (R("output/final_fleetA_sw09/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
for lo, hi in ((0.98, 1.01), (0.9, 0.98)):
    x = sc.filter((pl.col("p2") > lo) & (pl.col("p2") <= hi)).join(links, on=["si", "ti"], how="anti")
    x = x.join(links.rename({"si": "holder"}), on="ti", how="left").with_columns(pl.col("holder").is_null().alias("free"))
    x = x.join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti")
    print(f"US/India p2 in ({lo}, {hi}] not accepted: {x.height:,}", x.group_by("country", "free", "noaddr").len().sort("country", "free", "noaddr").rows())
