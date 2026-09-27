"""Cross-validated decisions (output/cv_TAG.parquet, blend p2 with the hybrid decode) on address-less true targets:
recall by k (Source 1 entities of the target's country sharing its name core), and what the missed ones look like."""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "src")
from amzn.train_v3 import decode  # noqa: E402
from amzn.train_v4 import one_to_one  # noqa: E402

tag = sys.argv[1] if len(sys.argv) > 1 else "pre1"
R = lambda p, c: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
cv = pl.read_parquet(f"output/cv_{tag}.parquet")
s1 = R("cache/train_source1.tsv", ["country", "name_core", "name"]).with_row_index("si")
tg = pl.concat([R(f"cache/train_source{i}.tsv", ["country", "address", "name_core", "name"]) for i in (2, 3)]).with_row_index("ti")
k = s1.group_by("country", "name_core").len().rename({"len": "k"})
tg = tg.join(k, on=["country", "name_core"], how="left").with_columns(pl.col("k").fill_null(0))
tg = tg.with_columns(pl.when(pl.col("k") >= 5).then(5).otherwise(pl.col("k")).alias("kb"), (pl.col("address") == "").alias("noaddr"))

scored = one_to_one(cv.select("si", "ti", "p2"), "p2")
chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p2"].to_numpy(), "expected", 0.05, 0.0)
picked = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                       "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
    pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.lit(True).alias("c"))
h = cv.join(picked, on=["si", "ti"], how="left").with_columns(pl.col("c").fill_null(False))
h = h.join(tg.select("ti", "kb", "noaddr"), on="ti")
true = h.filter(pl.col("label") == 1)
pl.Config.set_tbl_rows(30)
print("true pairs in CV candidates by address-less and k: recall of the decode")
print(true.group_by("noaddr", "kb").agg(pl.len().alias("n"), pl.col("c").mean().round(4).alias("recall")).sort("noaddr", "kb"))
# Where did the missed address-less targets go: assigned to another entity (wrong), or left unassigned?
missed = true.filter(pl.col("noaddr") & ~pl.col("c")).select("si", "ti", "kb", "p2")
taken = picked.select(pl.col("ti"), pl.col("si").alias("other"))
missed = missed.join(taken, on="ti", how="left")
print("missed address-less true targets:", missed.height, "of which assigned to another entity:", missed["other"].is_not_null().sum())
print(missed.group_by("kb").agg(pl.len(), pl.col("other").is_not_null().mean().round(3).alias("taken_by_other"),
                                pl.col("p2").median().alias("p2_med")).sort("kb"))
# Candidates per missed target: how many entities in CV compete for it, and the true owner's rank.
comp = h.filter(pl.col("noaddr")).join(missed.select("ti"), on="ti", how="semi")
rank = comp.with_columns(pl.col("p2").rank("ordinal", descending=True).over("ti").alias("r"),
                         pl.len().over("ti").alias("m")).filter(pl.col("label") == 1)
print("true owner's rank among candidate entities (missed targets):", rank.group_by("r").len().sort("r").head(6).rows())
print("candidate entities per missed target:", rank["m"].describe().rows())
# Precision of the decode on address-less targets it did assign.
pick_na = h.filter(pl.col("noaddr") & pl.col("c"))
print("address-less picks:", pick_na.height, "precision:", round(pick_na["label"].mean(), 4))
# Sample missed k=0 examples with names.
ex = missed.filter(pl.col("kb") == 0).head(12).join(s1.select("si", pl.col("name").alias("s_name")), on="si").join(
    tg.select("ti", pl.col("name").alias("t_name")), on="ti")
for r in ex.iter_rows(named=True):
    print(f"  p2={r['p2']:.3f} other={r['other']} | {r['s_name'][:40]:40} || {r['t_name'][:40]}")
