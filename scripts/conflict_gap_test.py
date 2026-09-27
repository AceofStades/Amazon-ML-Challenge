"""Test exact-name, same-street house conflicts by numeric gap: candidates per entity, stage-1 p1 and acceptance in a
submission, per country, next to the training true rate of the same gap bin."""
import re

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
x = pl.read_parquet("cache/conflict_test.parquet").filter(pl.col("name_exact") == 1)
hs = R("cache/test_source1.tsv", ["house"])["house"].fill_null("")
addr = pl.concat([R(f"cache/test_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
x = x.with_columns(hs.gather(x["si"].to_numpy()).alias("hs"), addr.gather(x["ti"].to_numpy()).alias("ta"))


def gap(h, ta):
    try:
        hv = int(h)
    except ValueError:
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta) if len(n) <= 5]
    if not nums:
        return None
    return min(abs(n - hv) for n in nums)


x = x.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
bins = [0, 2, 4, 6, 10, 20, 50, 100, 200, 500, 1000]
x = x.with_columns(pl.col("gap").cut(bins).alias("gb"))
x.select("si", "ti", "country", "gap", "p1", "acc").write_parquet("cache/conflict_gap_test.parquet")
E = {"us": 663106, "india": 809986, "france": 259452}
t = x.group_by("country", "gb").agg(pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc"),
                                    pl.col("p1").median().round(3).alias("p1_med"))
t = t.with_columns((pl.col("n") / pl.col("country").replace_strict(E, return_dtype=pl.Float64)).round(4).alias("per_E"))
tr = pl.read_parquet("cache/conflict_truth_train.parquet").filter(pl.col("name_exact") == 1)
trs = R("cache/train_source1.tsv", ["house"])["house"].fill_null("")
tra = pl.concat([R(f"cache/train_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
tr = tr.with_columns(trs.gather(tr["si"].to_numpy()).alias("hs"), tra.gather(tr["ti"].to_numpy()).alias("ta"))
tr = tr.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
tr = tr.with_columns(pl.col("gap").cut(bins).alias("gb")).group_by("gb").agg(
    (pl.len() / 2206821).round(4).alias("train_per_E"), pl.col("label").mean().round(3).alias("train_true"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(200)
wide = t.pivot(on="country", index="gb", values=["per_E", "acc", "p1_med"]).join(tr, on="gb", how="left").sort("gb")
print(wide)
