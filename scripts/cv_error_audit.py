"""Where the remaining cross-validated loss sits (test-like view: targets owned outside the CV entities removed).
Decodes a CV score file (si, ti, label, p_new) like the hybrid, then attributes every entity's F0.5 loss to its
missed links (by target type: address-less or not; unassigned or taken by another entity) and wrong links (target
owned by another CV entity, or by nobody)."""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from amzn.train_v3 import decode  # noqa: E402
from amzn.train_v4 import one_to_one  # noqa: E402

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
path, col = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "p_new")
full = pl.read_parquet(path).select("si", "ti", "label", pl.col(col).alias("p"))
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").select(pl.col("matched_entity_ids").alias("t")))
tid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
tid = tid.with_columns(pl.col("ti").cast(pl.UInt32), (pl.col("business_address") == "").alias("noaddr"))
has_owner = tid.join(gt, left_on="entity_id", right_on="t", how="semi").select("ti")
owned_in = full.filter(pl.col("label") == 1).select("ti").unique()
full = full.join(has_owner.join(owned_in, on="ti", how="anti"), on="ti", how="anti")
entities = full["si"].n_unique()
scored = one_to_one(full.select("si", "ti", "p"), "p")
chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p"].to_numpy(), "expected", 0.05, 0.0)
picked = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                       "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
    pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.lit(True).alias("c"))
h = full.join(picked, on=["si", "ti"], how="left").with_columns(pl.col("c").fill_null(False)).join(tid.select("ti", "noaddr"), on="ti")
g = h.group_by("si").agg(pl.col("label").sum().alias("a"), pl.col("c").sum().alias("k"), (pl.col("label").cast(pl.Boolean) & pl.col("c")).sum().alias("tp"))
a, k, tp = g["a"].to_numpy(), g["k"].to_numpy(), g["tp"].to_numpy()
f = np.where((a == 0) & (k == 0), 1.0, np.where((a == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * a + k, 1e-9)))
print(f"macro F0.5 {(f.sum() + entities - len(g)) / entities:.5f} over {entities:,} entities; loss {(len(g) - f.sum()) / entities:.5f}")
g = g.with_columns(pl.Series("loss", 1 - f))
# attribute each entity's loss evenly to its errors
fn = h.filter((pl.col("label") == 1) & ~pl.col("c"))
taken = picked.select("ti", pl.col("si").alias("winner"))
fn = fn.join(taken, on="ti", how="left").with_columns(
    pl.when(pl.col("winner").is_null()).then(pl.lit("unassigned")).otherwise(pl.lit("taken")).alias("how"))
fp = h.filter((pl.col("label") == 0) & pl.col("c"))
fp = fp.join(full.filter(pl.col("label") == 1).select("ti", pl.col("si").alias("owner")), on="ti", how="left").with_columns(
    pl.when(pl.col("owner").is_null()).then(pl.lit("no_owner")).otherwise(pl.lit("other_owner")).alias("how"))
err = pl.concat([fn.select("si", pl.lit("FN").alias("kind"), "noaddr", "how", "p"), fp.select("si", pl.lit("FP").alias("kind"), "noaddr", "how", "p")])
n_err = err.group_by("si").len().rename({"len": "n_err"})
err = err.join(n_err, on="si").join(g.select("si", "loss"), on="si").with_columns((pl.col("loss") / pl.col("n_err")).alias("share"))
t = err.group_by("kind", "noaddr", "how").agg(pl.len().alias("n"), (pl.col("share").sum() / entities).round(5).alias("loss"),
                                              pl.col("p").median().round(3).alias("p_med"))
print(t.sort("loss", descending=True))
err.write_parquet("cache/cv_errors.parquet")
