import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")


def links(p):
    return (R(p).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
            .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))


add = links("output/final_pre1_frc/matching_results.tsv").join(links("output/final_pre1/matching_results.tsv"), on=["si", "ti"], how="anti")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"])
x = add.join(sc, on=["si", "ti"], how="left")
print("restored", x.height)
print(x.with_columns(pl.col("p1").cut([0.1, 0.5]).alias("p1b"), pl.col("p2").cut([0.1, 0.5]).alias("p2b")).group_by("p1b", "p2b").len().sort("p1b", "p2b").rows())
