"""France free-target pairs by 6-model blend (pre1) p2 band, excluding swaps/brands/near added-word conflicts, with
agreement from the 22-model blend (fleetA) and stage 1; base = file 13 (France identical to file 11)."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
a = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"]).join(fr, on="si", how="semi").filter(pl.col("p2") > 0.5)
b = pl.read_parquet("output/blend_fleetA/test_scored.parquet", columns=["si", "ti", "p2"]).rename({"p2": "p2f"})
links = (R("output/final_safe_own/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
x = a.join(links.select("ti").unique(), on="ti", how="anti").join(b, on=["si", "ti"], how="left")
gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_swap_common", "g_brand1", "g_multi"])
cls = pl.read_parquet("cache/class_audit_fr.parquet", columns=["si", "ti", "h", "nd"])
x = x.join(gen, on=["si", "ti"], how="left").join(cls, on=["si", "ti"], how="left")
bad = (pl.col("g_brand1") == 1) | (pl.col("g_swap_common") == 1) | ((pl.col("h") == "near") & pl.col("nd").is_in(["add", "adddrop"]))
x = x.filter(~bad.fill_null(False))
x = x.with_columns(pl.col("p2").cut([0.7, 0.9, 0.98]).alias("band"), ((pl.col("p2f") > 0.9) & (pl.col("p1") > 0.5)).alias("agree"))
print(x.group_by("band", "agree").agg(pl.len(), pl.col("ti").n_unique().alias("targets")).sort("band", "agree").rows())
print("classes in (0.9, 0.98] with agreement:", x.filter((pl.col("band") == "(0.9, 0.98]") & pl.col("agree")).group_by(
    pl.col("h").fill_null("other"), pl.col("nd").fill_null("-")).len().sort("len", descending=True).head(8).rows())
x.write_parquet("cache/fr_band.parquet")
