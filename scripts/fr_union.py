"""France blend union candidates: France pairs with blend p2 above a threshold that the current submission does not
accept, by class and by target state (free / held by another entity), across all France candidate pairs."""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
sub, thr = sys.argv[1], float(sys.argv[2])
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"]).join(fr, on="si", how="semi")
links = (R(sub).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
hi = sc.filter(pl.col("p2") > thr).join(links, on=["si", "ti"], how="anti")
holder = links.rename({"si": "holder"})
hi = hi.join(holder, on="ti", how="left")
cls = pl.read_parquet("cache/class_audit_fr.parquet").select("si", "ti", "h", "nd")
hi = hi.join(cls, on=["si", "ti"], how="left").with_columns(pl.col("h").fill_null("other"), pl.col("nd").fill_null("-"))
gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_acronym", "g_domain", "g_brand1", "g_alias", "g_typo1", "g_swap_common"])
hi = hi.join(gen, on=["si", "ti"], how="left").join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti")
hi = hi.with_columns(pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).otherwise(pl.col("h")).alias("h"))
hi = hi.with_columns(pl.when(pl.col("holder").is_null()).then(pl.lit("free")).otherwise(pl.lit("held")).alias("state"))
pl.Config.set_tbl_rows(60)
print(f"France pairs with p2 > {thr} not accepted: {hi.height:,}")
print(hi.group_by("h", "nd", "state").len().pivot(on="state", index=["h", "nd"], values="len").fill_null(0).sort("h", "nd"))
o = hi.filter(pl.col("h") == "other")
print("other (not same-street similar-name):", o.select(pl.col("g_acronym").sum(), pl.col("g_domain").sum(), pl.col("g_brand1").sum(),
                                                        pl.col("g_alias").sum(), pl.col("g_typo1").sum(), pl.col("g_swap_common").sum()).rows())
hi.write_parquet("cache/fr_union.parquet")
