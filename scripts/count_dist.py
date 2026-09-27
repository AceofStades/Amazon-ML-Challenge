"""Distribution of predicted matches per Source 1 entity by country (a submission) against the training truth."""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1t = R("dataset/train/train_source1.tsv", ["entity_id", "country"])
gt = R("dataset/train/train_ground_truth.tsv").join(s1t, left_on="source1_entity_id", right_on="entity_id")
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"])
sub = R(sys.argv[1]).join(s1, left_on="source1_entity_id", right_on="entity_id")


def dist(frame, tag):
    f = frame.with_columns(pl.when(pl.col("matched_entity_ids") == "").then(0).otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("k"))
    f = f.with_columns(pl.when(pl.col("k") >= 8).then(8).otherwise(pl.col("k")).alias("kb"))
    t = f.group_by("country", "kb").len().with_columns((pl.col("len") / pl.col("len").sum().over("country")).round(4).alias("share"))
    print(tag)
    print(t.pivot(on="country", index="kb", values="share").sort("kb"))


pl.Config.set_tbl_rows(20)
dist(gt, "train truth")
dist(sub, "test " + sys.argv[1])
