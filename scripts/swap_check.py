"""Common-word swaps in France (one word replaced by a frequent France word) by house relation: same house versus
other houses of the same street, their blend p2, and current acceptance. If France swap decoys, like France's
added-word decoys, avoid the Source 1 house, same-house swaps are true records and the swap-drop rule was wrong."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", "g_swap_common", "g_swap_inplace", "g_swap_moved", "g_swap_filler", "g_swap_rare"]).join(fr, on="si", how="semi")
sw = gen.filter((pl.col("g_swap_common") == 1) | (pl.col("g_swap_rare") == 1) | (pl.col("g_swap_filler") == 1))
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos"]
f = pl.concat([pl.read_parquet(p, columns=cols).join(sw.select("si", "ti"), on=["si", "ti"], how="semi") for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
sw = sw.join(f, on=["si", "ti"], how="left")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"])
sw = sw.join(sc, on=["si", "ti"], how="left")
links = (R("output/final_pre1_frcu/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("acc")))
sw = sw.join(links, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
sw = sw.with_columns(pl.when(pl.col("g_swap_common") == 1).then(pl.lit("common")).when(pl.col("g_swap_filler") == 1).then(pl.lit("filler")).otherwise(pl.lit("rare")).alias("kind"),
                     pl.when(pl.col("address_idf_cos") < 0.6).then(pl.lit("otherst")).when(pl.col("house_match") == 1).then(pl.lit("same"))
                     .when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")).alias("where"))
t = sw.group_by("kind", "where").agg(pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc"), (pl.col("p2") > 0.98).mean().round(3).alias("p2_hi"),
                                     (pl.col("p2") < 0.1).mean().round(3).alias("p2_lo"))
pl.Config.set_tbl_rows(30)
print(t.sort("kind", "where"))
