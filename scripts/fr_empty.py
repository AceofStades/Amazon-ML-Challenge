"""France entities predicted empty: their best candidate (stage-1 p1, 6-model blend p2), whether that target is held
by another entity, and its class; compared with the same view for US on a submission."""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
m = R(sub)
empty = m.filter(pl.col("matched_entity_ids") == "").join(s1, left_on="source1_entity_id", right_on="entity_id").select("si", "country")
links = (m.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"]).join(empty.select("si"), on="si", how="semi")
best = sc.sort("p1", descending=True).unique("si", keep="first").join(empty, on="si")
best = best.join(links.select("ti", pl.lit(1).alias("held")), on="ti", how="left").with_columns(pl.col("held").fill_null(0))
best = best.join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti")
E = empty.group_by("country").len()
print("predicted-empty entities:", E.sort("country").rows())
print(best.with_columns(pl.col("p1").cut([0.1, 0.3, 0.5, 0.7, 0.9]).alias("b")).group_by("country", "b", "held").len()
      .pivot(on="country", index=["b", "held"], values="len").fill_null(0).sort("b", "held"))
best.write_parquet("cache/fr_empty_best.parquet")
