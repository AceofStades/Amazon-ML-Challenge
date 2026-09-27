"""Consistency of exact duplicate Source 2/3 records (same country, normalized name and address).
Training: how often copies share the true owner. Test: how our submission treats copies.
Usage: python scripts/agent_dup_consistency.py SUBMISSION"""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False,
                                  missing_utf8_is_empty_string=True, columns=c)


def owners(path):
    m = R(path).rename(lambda c: "s1" if c.startswith("source1") else "ids")
    return (m.filter(pl.col("ids") != "").with_columns(pl.col("ids").str.split(","))
            .explode("ids").rename({"ids": "entity_id"}))


def audit(split, sub):
    tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["entity_id", "country", "name", "address"]) for i in (2, 3)])
    tg = tg.join(owners(sub), on="entity_id", how="left")
    tg = tg.with_columns((pl.col("address") == "").alias("noaddr"))
    g = (tg.group_by("country", "name", "address", "noaddr")
         .agg(pl.len().alias("n"), pl.col("s1").drop_nulls().n_unique().alias("n_own"),
              pl.col("s1").null_count().alias("n_free")))
    g = g.filter(pl.col("n") >= 2)
    g = g.with_columns(pl.when(pl.col("n_own") == 0).then(pl.lit("all_free"))
                       .when((pl.col("n_own") == 1) & (pl.col("n_free") == 0)).then(pl.lit("same_owner"))
                       .when((pl.col("n_own") == 1)).then(pl.lit("one_owner+free"))
                       .when(pl.col("n_free") == 0).then(pl.lit("diff_owners"))
                       .otherwise(pl.lit("diff_owners+free")).alias("kind"))
    t = (g.group_by("country", "noaddr", "kind").agg(pl.len().alias("groups"), pl.col("n").sum().alias("records"),
                                                      pl.col("n_free").sum().alias("free_recs"))
         .sort("country", "noaddr", "kind"))
    return t


if __name__ == "__main__":
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(160)
    print("TRAIN (truth)")
    print(audit("train", "dataset/train/train_ground_truth.tsv"))
    print("TEST", sys.argv[1])
    print(audit("test", sys.argv[1]))
