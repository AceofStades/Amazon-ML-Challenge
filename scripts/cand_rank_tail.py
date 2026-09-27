"""Candidate-list saturation: rank of true training links vs accepted test links within the per-entity candidate lists
(embed_rank: bi-encoder top 40; candidate_rank: key blocking top 30). A heavier tail near the cutoff in test means
the extra test decoys push true links out of the lists."""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def ranks(split, pairs):
    cols = ["si", "ti", "embed_rank", "candidate_rank", "in_keys", "in_knn"]
    parts = [pl.read_parquet(f, columns=cols).join(pairs, on=["si", "ti"], how="semi") for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))]
    return pl.concat(parts)


s1 = R("dataset/train/train_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
      .join(tid, left_on="matched_entity_ids", right_on="entity_id").select(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), "country"))
gt = gt.filter(pl.col("si").hash(3) % 20 == 0)
tr = ranks("train", gt.select("si", "ti")).join(gt, on=["si", "ti"])
s1t = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tgt = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
sub = (R(sys.argv[1]).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
       .join(s1t, left_on="source1_entity_id", right_on="entity_id").join(tgt, left_on="matched_entity_ids", right_on="entity_id")
       .select(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), "country"))
te = ranks("test", sub.select("si", "ti")).join(sub, on=["si", "ti"])
for name, f in (("train true", tr), ("test accepted", te)):
    g = f.group_by("country").agg(pl.len().alias("n"),
                                  (pl.col("embed_rank") >= 30).mean().round(4).alias("knn_rank>=30"),
                                  (pl.col("embed_rank") >= 36).mean().round(4).alias("knn_rank>=36"),
                                  (pl.col("candidate_rank") >= 20).mean().round(4).alias("key_rank>=20"),
                                  (pl.col("candidate_rank") >= 26).mean().round(4).alias("key_rank>=26"),
                                  (pl.col("in_knn") == 0).mean().round(4).alias("keys_only"),
                                  (pl.col("in_keys") == 0).mean().round(4).alias("knn_only"))
    print(name); print(g.sort("country"))
