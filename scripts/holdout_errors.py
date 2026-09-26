"""Holdout error audit for a blend: false negatives and false positives by category, their F0.5 cost, and examples."""
import glob
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from amzn.train_v3 import decode  # noqa: E402
from amzn.train_v4 import one_to_one  # noqa: E402

folder = sys.argv[1]
R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
hold = pl.read_parquet(f"{folder}/valid_predictions.parquet")
scored = one_to_one(hold.select("si", "ti", "p2"), "p2")
chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p2"].to_numpy(), "expected", 0.05, 0.0)
picked = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                       "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
    pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.lit(True).alias("acc"))
h = hold.join(picked, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(False))
# Truth links outside the candidates.
s_ids = pl.read_parquet("cache/embeddings/train_source1_ids.parquet").select("entity_id", "country").with_row_index("si")
truth = (R("dataset/train/train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
         .filter(pl.col("matched_entity_ids") != "").join(s_ids, left_on="source1_entity_id", right_on="entity_id")
         .join(h.select("si").unique(), on="si", how="semi"))
print(f"holdout entities with candidates {h['si'].n_unique():,}; true links {truth.height:,}; in candidates {int(h['label'].sum()):,}")
feat_cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos", "address_missing", "name_missing",
             "s_name_dup", "t_name_dup", "s_name_in_targets"]
feat = pl.concat([pl.read_parquet(p, columns=feat_cols).join(h.select("si", "ti"), on=["si", "ti"], how="semi")
                  for p in sorted(glob.glob("cache/v3/train/part*.parquet"))])
h = h.join(feat, on=["si", "ti"], how="left").join(s_ids.select("si", "country"), on="si")
fn = h.filter((pl.col("label") == 1) & ~pl.col("acc"))
fp = h.filter((pl.col("label") == 0) & pl.col("acc"))
print(f"FN {fn.height:,}  FP {fp.height:,}")


def cat(frame):
    return frame.with_columns(
        pl.when(pl.col("address_missing") == 1).then(pl.lit("t_noaddr"))
        .when(pl.col("house_match") == 1).then(pl.lit("house_eq"))
        .when(pl.col("house_conflict") == 1).then(pl.lit("house_cf")).otherwise(pl.lit("house_un")).alias("H"),
        pl.when(pl.col("name_idf_cos") >= 0.95).then(pl.lit("name_same")).when(pl.col("name_idf_cos") >= 0.6)
        .then(pl.lit("name_close")).otherwise(pl.lit("name_far")).alias("N"),
        pl.when(pl.col("s_name_dup") > 1).then(pl.lit("s_dup")).otherwise(pl.lit("s_uniq")).alias("D"))


for name, frame in (("FN", fn), ("FP", fp)):
    c = cat(frame)
    print(f"\n{name} by category:")
    print(c.group_by("H", "N", "D").len().sort("len", descending=True).head(12).rows())
    print(f"{name} by country:", c.group_by("country").len().rows(), " p2 quartiles:", [round(c["p2"].quantile(q), 3) for q in (0.25, 0.5, 0.75)])
s1 = R("dataset/train/train_source1.tsv").with_row_index("si").select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa"))
tg = pl.concat([R(f"dataset/train/train_source{i}.tsv") for i in (2, 3)]).with_row_index("ti").select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta"))
for name, frame in (("FN", fn), ("FP", fp)):
    print(f"\n{name} examples:")
    for r in cat(frame).sample(min(18, frame.height), seed=3).join(s1, on="si").join(tg, on="ti").iter_rows(named=True):
        print(f"  p2={r['p2']:.2f} {r['H']:8} {r['N']:10} dup={r['s_name_dup']:.0f} | {r['sn'][:28]:28} | {r['sa'][:30]:30} || {r['tn'][:28]:28} | {r['ta'][:32]}")
