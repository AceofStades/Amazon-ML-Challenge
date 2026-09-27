"""Training: are there decoys with the exact Source 1 name at the Source 1 house? True rate of exact-name pairs by house
relation (same house, conflict, unknown), to size the decoy placement ratio for exact names."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
cols = ["source1_entity_id", "candidate_entity_id", "house_match", "house_conflict", "address_idf_cos", "name_exact", "name_idf_cos"]
x = pl.concat([pl.read_parquet(f, columns=cols).filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.95))
               for f in sorted(glob.glob("cache/v3/train/part*.parquet"))])
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").select("source1_entity_id", pl.col("matched_entity_ids").alias("candidate_entity_id")).with_columns(pl.lit(1).alias("label")))
x = x.join(gt, on=["source1_entity_id", "candidate_entity_id"], how="left").with_columns(pl.col("label").fill_null(0))
x = x.with_columns(pl.when(pl.col("house_match") == 1).then(pl.lit("same_house")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict"))
                   .otherwise(pl.lit("unknown")).alias("hrel"))
print(x.group_by("hrel", "name_exact").agg(pl.len().alias("n"), (pl.len() / 2206821).round(4).alias("per_E"),
                                           pl.col("label").mean().round(4).alias("true"), (pl.col("label") == 0).sum().alias("decoys")).sort("hrel", "name_exact"))
