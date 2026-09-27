"""Source-balance scan of every unassigned France candidate class (free target, stage-1 p1 >= 0.2 or blend p2 >= 0.5):
house relation x name-change kind x score band. pi = (fill - 0.5) / (ref - 0.5), ref = fill of France accepted exact
links under the same conditioning (entity's accepted links, 2 or 3, all in one source).
Usage: python scripts/fr_pool_balance.py SUBMISSION"""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
KINDS = ["g_alias", "g_domain", "g_acronym", "g_exact", "g_typo1", "g_drop", "g_add", "g_swap_filler", "g_swap_common", "g_swap_rare", "g_multi", "g_brand1"]
sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
tg = tg.with_columns(pl.col("entity_id").str.starts_with("S3").cast(pl.Int32).alias("t3"), (pl.col("business_address") == "").alias("noaddr"))
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti", "t3"), left_on=col, right_on="entity_id").select("si", "ti", "t3", "country"))
held = l.select("ti").unique()
fr_l = l.filter(pl.col("country") == "France")
ent = fr_l.group_by("si").agg(pl.len().alias("k"), pl.col("t3").sum().alias("n3"))
ent = ent.filter(pl.col("k").is_in([2, 3]) & ((pl.col("n3") == 0) | (pl.col("n3") == pl.col("k"))))
g = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti"] + KINDS)
ex = fr_l.join(g.filter(pl.col("g_exact") > 0.5).select("si", "ti"), on=["si", "ti"], how="semi")
e2 = fr_l.group_by("si").agg(pl.len().alias("K"), pl.col("t3").sum().alias("N3"))
ex = ex.join(e2, on="si").filter(pl.col("K").is_in([3, 4])).with_columns((pl.col("K") - 1).alias("kr"), (pl.col("N3") - pl.col("t3")).alias("n3r"))
ex = ex.filter((pl.col("n3r") == 0) | (pl.col("n3r") == pl.col("kr")))
ref = ex.with_columns(pl.when(pl.col("n3r") == 0).then(pl.col("t3") == 1).otherwise(pl.col("t3") == 0).alias("fill"))["fill"].mean()
p1 = pl.concat([pl.read_parquet(f).join(ent.select("si"), on="si", how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
sc = pl.read_parquet("output/blend_fleetZ/test_scored.parquet", columns=["si", "ti", "p2"]).join(ent.select("si"), on="si", how="semi")
c = p1.join(held, on="ti", how="anti").join(sc, on=["si", "ti"], how="left").with_columns(pl.col("p2").fill_null(0.0))
c = c.filter((pl.col("p1") >= 0.2) | (pl.col("p2") >= 0.5))
f = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos"]).join(c.select("si", "ti"), on=["si", "ti"], how="semi")
               for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
c = c.join(f, on=["si", "ti"], how="left").join(g, on=["si", "ti"], how="left").join(tg.select("ti", "t3", "noaddr"), on="ti")
kind = pl.coalesce([pl.when(pl.col(k) > 0.5).then(pl.lit(k[2:])) for k in KINDS] + [pl.lit("other")])
house = (pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when(pl.col("address_idf_cos").is_null() | (pl.col("address_idf_cos") < 0.6)).then(pl.lit("otherst"))
         .when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")))
band = pl.when(pl.col("p2") >= 0.95).then(pl.lit("p2>=.95")).when(pl.col("p2") >= 0.5).then(pl.lit("p2 .5-.95")).otherwise(pl.lit("p2<.5"))
c = c.with_columns(kind.alias("kind"), house.alias("house"), band.alias("band")).join(ent, on="si")
c = c.with_columns(pl.when(pl.col("n3") == 0).then(pl.col("t3") == 1).otherwise(pl.col("t3") == 0).alias("fill"))
t = c.group_by("house", "kind", "band").agg(pl.len().alias("n"), pl.col("fill").mean().alias("fill"))
t = t.with_columns(((pl.col("fill") - 0.5) / (ref - 0.5)).round(2).alias("pi"), (0.5 / (pl.col("n").sqrt() * (ref - 0.5))).round(2).alias("se"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
print(f"reference fill {ref:.3f}; sample = entities with 2-3 accepted one-source links")
print(t.filter(pl.col("n") >= 100).sort("pi", descending=True).head(40))
