"""Per-entity split of true links across Source 2 and Source 3, and records without an address."""
import polars as pl

R = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
g = (R("dataset/train/train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.slice(0, 2).alias("src")))
per = g.group_by("source1_entity_id").agg((pl.col("src") == "S2").sum().alias("n2"), (pl.col("src") == "S3").sum().alias("n3"))
per = per.with_columns((pl.col("n2") + pl.col("n3")).alias("n"))
pl.Config.set_tbl_rows(30)
print(per.group_by("n").agg(pl.len(), (pl.col("n2") == 0).mean().round(3).alias("no_S2"), (pl.col("n3") == 0).mean().round(3).alias("no_S3"),
                            pl.col("n2").mean().round(2).alias("mean_n2")).sort("n").head(10))
print("top (n2, n3):", per.group_by("n2", "n3").len().sort("len", descending=True).head(15).rows())
t = pl.concat([R(f"dataset/train/train_source{i}.tsv").select("entity_id", "business_address") for i in (2, 3)])
g2 = g.join(t, left_on="matched_entity_ids", right_on="entity_id").with_columns((pl.col("business_address") == "").alias("noaddr"))
pa = g2.group_by("source1_entity_id").agg(pl.col("noaddr").sum().alias("k"), pl.len().alias("n"))
print("true records without address per entity:", pa.group_by("k").len().sort("k").head(6).rows(), "share of links:", round(g2["noaddr"].mean(), 4))
print("no-address records by source:", g2.filter(pl.col("noaddr")).group_by("src").len().rows())
