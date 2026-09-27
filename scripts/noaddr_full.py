"""Full-competition view on training (every Source 1 entity, out-of-fold stage-1 p1): for each address-less target
that is a true link, the owner's rank among all candidate entities by p1, and how well "assign to the top candidate"
would do by top-p1 and margin cells. Saves cache/noaddr_train.parquet (per target: top1/top2 p1, owner flags)."""
import glob

import polars as pl

R = lambda p, c: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
tg = pl.concat([R(f"cache/train_source{i}.tsv", ["entity_id", "address"]) for i in (2, 3)]).with_row_index("ti")
s1 = R("cache/train_source1.tsv", ["entity_id"]).with_row_index("si")
noaddr = tg.filter(pl.col("address") == "").select("ti", pl.col("entity_id").alias("t_id"))
parts = []
for f in sorted(glob.glob("output/v7/train_p1/*.parquet")):
    parts.append(pl.read_parquet(f).join(noaddr.select("ti"), on="ti", how="semi"))
p = pl.concat(parts)
gt = (R("dataset/train/train_ground_truth.tsv", None).filter(pl.col("matched_entity_ids") != "")
      .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
      .select(pl.col("source1_entity_id").alias("s_id"), pl.col("matched_entity_ids").alias("t_id")))
gt = gt.join(noaddr, on="t_id").join(s1.rename({"entity_id": "s_id"}), on="s_id").select("si", "ti").with_columns(pl.lit(1).alias("label"))
p = p.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
print("address-less targets:", noaddr.height, "with candidates:", p["ti"].n_unique(), "true targets:", gt.height,
      "true pairs in candidates:", p["label"].sum())
p = p.with_columns(pl.col("p1").rank("ordinal", descending=True).over("ti").alias("r"))
top = p.group_by("ti").agg(pl.col("p1").max().alias("top1"), pl.col("p1").sort(descending=True).get(1, null_on_oob=True).alias("top2"),
                           pl.len().alias("m"), (pl.col("label") * (pl.col("r") == 1)).max().alias("top_is_owner"),
                           pl.col("label").max().alias("has_owner"))
top = top.with_columns(pl.col("top2").fill_null(0.0))
top.write_parquet("cache/noaddr_train.parquet")
top = top.with_columns(pl.col("top1").cut([0.1, 0.2, 0.3, 0.5, 0.7, 0.9]).alias("t1"),
                       (pl.col("top2") / pl.col("top1").clip(1e-6)).cut([0.1, 0.25, 0.5, 0.75]).alias("ratio"))
pl.Config.set_tbl_rows(60)
print(top.group_by("t1", "ratio").agg(pl.len().alias("n"), pl.col("top_is_owner").mean().round(3).alias("prec_top"))
      .sort("t1", "ratio"))
