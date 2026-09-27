"""Address-less targets: how many Source 1 entities of the same country share the target's normalized name core (k),
split by whether the submission assigned the target. Test uses a submission file; train uses the ground truth."""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def links(path):
    return (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").select(pl.col("source1_entity_id").alias("s"), pl.col("matched_entity_ids").alias("t")))


def run(split, sub):
    s1 = R(f"cache/{split}_source1.tsv", ["entity_id", "country", "name_core"])
    tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["entity_id", "country", "address", "name_core"]) for i in (2, 3)])
    na = tg.filter(pl.col("address") == "")
    k = s1.group_by("country", "name_core").len().rename({"len": "k"})
    na = na.join(k, on=["country", "name_core"], how="left").with_columns(pl.col("k").fill_null(0))
    lk = links(sub)
    na = na.join(lk, left_on="entity_id", right_on="t", how="left").with_columns(pl.col("s").is_not_null().alias("assigned"))
    na = na.with_columns(pl.when(pl.col("k") >= 5).then(pl.lit(5)).otherwise(pl.col("k")).alias("kb"))
    pl.Config.set_tbl_rows(40)
    print(split, sub)
    print(na.group_by("country", "kb").agg(pl.len().alias("n"), pl.col("assigned").mean().round(3).alias("assigned_share"))
          .sort("country", "kb"))


run("train", "dataset/train/train_ground_truth.tsv")
run("test", sys.argv[1])
