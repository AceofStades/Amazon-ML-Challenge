import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")


def links(p):
    return (R(p).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
            .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))


add = links("output/final_pre1_frcu/matching_results.tsv").join(links("output/final_pre1_frc/matching_results.tsv"), on=["si", "ti"], how="anti")
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
f = pl.concat([pl.read_parquet(p, columns=cols).join(add, on=["si", "ti"], how="semi") for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
f = f.join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti")
f = f.with_columns(pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when(pl.col("address_idf_cos") >= 0.6).then(pl.lit("street")).otherwise(pl.lit("otherst")).alias("st"),
                   pl.when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")).alias("hr"),
                   (pl.col("name_idf_cos") >= 0.8).alias("simname"))
print(f.group_by("st", "hr", "simname").len().sort("len", descending=True).rows())
