"""France entities with exactly one predicted match: their best unclaimed other candidate by stage-1 p1, its class and
house relation, and the 6-model blend's view; the same counts for US as reference."""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
m = R(sys.argv[1])
links = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti", "country"))
one = links.group_by("si", "country").len().filter(pl.col("len") == 1).select("si", "country")
print("one-match entities:", one.group_by("country").len().sort("country").rows())
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"]).join(one.select("si"), on="si", how="semi")
sc = sc.join(links.select("si", "ti"), on=["si", "ti"], how="anti").join(links.select("ti").unique(), on="ti", how="anti")
best = sc.sort("p1", descending=True).unique("si", keep="first").join(one, on="si")
gen = pl.read_parquet("cache/gen/test.parquet")
cls = pl.read_parquet("cache/class_audit_fr.parquet", columns=["si", "ti", "h", "nd"])
classes = [("acronym", "g_acronym"), ("domain", "g_domain"), ("alias", "g_alias"), ("brand1", "g_brand1"), ("swap_common", "g_swap_common"),
           ("swap_filler", "g_swap_filler"), ("swap_rare", "g_swap_rare"), ("typo1", "g_typo1"), ("exact", "g_exact"), ("add", "g_add"), ("drop", "g_drop"), ("multi", "g_multi")]
lab = pl.lit("other")
for name, col in reversed(classes):
    lab = pl.when(pl.col(col) == 1).then(pl.lit(name)).otherwise(lab)
best = best.join(gen, on=["si", "ti"], how="left").join(cls, on=["si", "ti"], how="left").join(
    tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti").with_columns(lab.alias("cls"), pl.col("h").fill_null("-"))
f = best.filter((pl.col("country") == "France") & (pl.col("p1") > 0.5))
print("France one-match entities with a free candidate p1 > 0.5:", f.height)
print(f.group_by("cls", "h", "noaddr").agg(pl.len(), (pl.col("p2") > 0.5).mean().round(2).alias("p2>.5")).sort("len", descending=True).head(14).rows())
u = best.filter((pl.col("country") == "US") & (pl.col("p1") > 0.5))
print("US one-match entities with a free candidate p1 > 0.5:", u.height)
