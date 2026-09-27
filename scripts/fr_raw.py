"""France pairs that stage 1 scores high (p1 >= 0.8) but the current France decisions reject, on targets no entity
holds, after removing the decoy classes identified on the leaderboard (common-word swaps, made-up brands, near
added-word conflicts) and multi-word changes: counts by class with the 6-model blend's opinion."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"]).join(fr, on="si", how="semi").filter(pl.col("p1") >= 0.8)
links = (R("output/final_safe_own/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
x = sc.join(links, on=["si", "ti"], how="anti")
x = x.join(links.select("ti").unique().with_columns(pl.lit(1).alias("held")), on="ti", how="left").with_columns(pl.col("held").fill_null(0))
gen = pl.read_parquet("cache/gen/test.parquet")
cls = pl.read_parquet("cache/class_audit_fr.parquet", columns=["si", "ti", "h", "nd"])
x = x.join(gen, on=["si", "ti"], how="left").join(cls, on=["si", "ti"], how="left").join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti")
classes = [("acronym", "g_acronym"), ("domain", "g_domain"), ("alias", "g_alias"), ("swap_common", "g_swap_common"), ("brand1", "g_brand1"),
           ("swap_filler", "g_swap_filler"), ("swap_rare", "g_swap_rare"), ("typo1", "g_typo1"), ("exact", "g_exact"), ("add", "g_add"), ("drop", "g_drop"), ("multi", "g_multi")]
lab = pl.lit("other")
for name, col in reversed(classes):
    lab = pl.when(pl.col(col) == 1).then(pl.lit(name)).otherwise(lab)
x = x.with_columns(lab.alias("cls"))
print("France p1 >= 0.8 rejected:", x.height, "free targets:", (x["held"] == 0).sum())
bad = pl.col("cls").is_in(["swap_common", "brand1", "multi"]) | ((pl.col("h") == "near") & pl.col("nd").is_in(["add", "adddrop"]))
f = x.filter((pl.col("held") == 0) & ~bad.fill_null(False))
print("free, decoy classes removed:", f.height, "unique targets:", f["ti"].n_unique())
t = f.group_by("cls", pl.col("noaddr")).agg(pl.len(), (pl.col("p2") > 0.5).mean().round(3).alias("p2>.5"), (pl.col("p2") < 0.05).mean().round(3).alias("p2<.05"),
                                            pl.col("p1").median().round(3).alias("p1_med"))
pl.Config.set_tbl_rows(40)
print(t.sort("len", descending=True))
f.write_parquet("cache/fr_raw_pool.parquet")
