"""France exact-name same-street conflicts: is the target already assigned to another Source 1 entity (typically a
same-name entity at the target's own house)? Only unassigned targets can be missed true links."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
te = pl.read_parquet("cache/conflict_fr.parquet")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
sub = (R("output/final_pre1/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
       .join(tg, left_on="matched_entity_ids", right_on="entity_id").select(pl.col("si").alias("owner"), "ti"))
te = te.join(sub, on="ti", how="left").with_columns(
    pl.when(pl.col("acc") == 1).then(pl.lit("accepted")).when(pl.col("owner").is_not_null()).then(pl.lit("other_entity")).otherwise(pl.lit("unassigned")).alias("state"),
    (pl.col("p1") >= 0.5).alias("p1_high"))
pl.Config.set_tbl_rows(60)
print(te.group_by("gb", "p1_high", "state").len().pivot(on="state", index=["gb", "p1_high"], values="len").sort("p1_high", "gb"))
te.filter(pl.col("p1_high") & (pl.col("state") == "unassigned")).select("si", "ti", "gap", "p1", "rel").write_parquet("cache/conflict_fr_unassigned_high.parquet")
