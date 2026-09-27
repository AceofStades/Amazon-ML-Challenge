"""Exact-name, same-street house conflicts that stage 1 already scores high (p1 >= 0.5): training true rate by gap bin,
against the France test count per bin and the share the current France decisions keep."""
import glob
import re

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
bins = [0, 2, 4, 6, 10, 20, 50, 100, 200, 500]


def gap(h, ta):
    try:
        hv = int(h)
    except ValueError:
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta) if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


tr = pl.read_parquet("cache/conflict_truth_train.parquet").filter(pl.col("name_exact") == 1)
p1 = pl.concat([pl.read_parquet(f).join(tr.select("si", "ti"), on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/train_p1/*.parquet"))])
tr = tr.join(p1, on=["si", "ti"], how="left")
trs = R("cache/train_source1.tsv", ["house"])["house"].fill_null("")
tra = pl.concat([R(f"cache/train_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
tr = tr.with_columns(trs.gather(tr["si"].to_numpy()).alias("hs"), tra.gather(tr["ti"].to_numpy()).alias("ta"))
tr = tr.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
tr = tr.with_columns(pl.col("gap").cut(bins).alias("gb"), pl.when(pl.col("rel") == "target_1").then(pl.lit("t1")).otherwise(pl.lit("")).alias("t1"))
a = tr.filter(pl.col("p1") >= 0.5).group_by("gb", "t1").agg(pl.len().alias("train_n"), pl.col("label").mean().round(3).alias("train_true"))
b = tr.group_by("gb", "t1").agg(pl.col("label").mean().round(3).alias("train_true_all"))

te = pl.read_parquet("cache/conflict_gap_test.parquet").filter(pl.col("country") == "france")
rel = pl.read_parquet("cache/conflict_test.parquet").select("si", "ti", "rel")
te = te.join(rel, on=["si", "ti"], how="left").with_columns(pl.col("gap").cut(bins).alias("gb"),
                                                           pl.when(pl.col("rel") == "target_1").then(pl.lit("t1")).otherwise(pl.lit("")).alias("t1"))
c = te.filter(pl.col("p1") >= 0.5).group_by("gb", "t1").agg(pl.len().alias("fr_n"), pl.col("acc").mean().round(3).alias("fr_acc"))
pl.Config.set_tbl_rows(40)
print(a.join(b, on=["gb", "t1"]).join(c, on=["gb", "t1"], how="full", coalesce=True).sort("t1", "gb"))
te.write_parquet("cache/conflict_fr.parquet")
