"""France follow-ups after file 11 (blend-confident France pairs proved true on the leaderboard):
1. free-target France pairs with 0.9 < p2 <= 0.98, by class (decoy classes excluded as in france_union.py);
2. switches: targets held by a France entity the blend rejects (p2 < 0.1) while it is sure (p2 > 0.98) of another."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
sc = pl.read_parquet("output/blend_fleetA/test_scored.parquet", columns=["si", "ti", "p1", "p2"]).join(fr, on="si", how="semi")
links = (R("output/final_fleetA_sw/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti")).join(fr, on="si", how="semi")
gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_swap_common", "g_brand1"])
cls = pl.read_parquet("cache/class_audit_fr.parquet", columns=["si", "ti", "h", "nd"])
mid = sc.filter((pl.col("p2") > 0.9) & (pl.col("p2") <= 0.98)).join(links.select("ti").unique(), on="ti", how="anti")
mid = mid.join(gen, on=["si", "ti"], how="left").join(cls, on=["si", "ti"], how="left")
bad = (pl.col("g_brand1") == 1) | ((pl.col("h") == "near") & pl.col("nd").is_in(["add", "adddrop"]))
mid = mid.filter(~bad.fill_null(False))
print(f"free France pairs with 0.9 < p2 <= 0.98 (no brands, no near added-word): {mid.height:,}; unique targets {mid['ti'].n_unique():,}; swaps {int(mid['g_swap_common'].fill_null(0).sum()):,}")
print("  by class:", mid.group_by(pl.col("h").fill_null("other"), pl.col("nd").fill_null("-")).len().sort("len", descending=True).head(8).rows())
held = links.join(sc.select("si", "ti", pl.col("p2").alias("p2_holder")), on=["si", "ti"], how="left").rename({"si": "holder"})
alt = sc.filter(pl.col("p2") > 0.98).join(held, on="ti").filter((pl.col("holder") != pl.col("si")) & (pl.col("p2_holder") < 0.1))
print(f"switch candidates (holder p2 < 0.1, other entity p2 > 0.98): {alt.height:,}")
