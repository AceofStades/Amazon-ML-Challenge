"""Training exact-name, same-street house conflicts: true rate by the numeric gap between the Source 1 house number
and the closest target number, and by relative gap, to see how decoys "a few doors away" differ from true records
whose number was randomized."""
import re

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
x = pl.read_parquet("cache/conflict_truth_train.parquet").filter(pl.col("name_exact") == 1)
hs = R("cache/train_source1.tsv", ["house"])["house"].fill_null("")
addr = pl.concat([R(f"cache/train_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
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


x = x.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"),
                   pl.col("hs").cast(pl.Int64, strict=False).alias("hv"))
x = x.with_columns(pl.col("gap").cut([0, 2, 4, 6, 10, 20, 50, 100, 200, 500, 1000]).alias("gb"),
                   (pl.col("gap") / pl.col("hv").clip(1)).cut([0.05, 0.1, 0.25, 0.5, 1, 2, 5]).alias("rb"))
pl.Config.set_tbl_rows(40)
print(x.group_by("gb").agg(pl.len(), pl.col("label").mean().round(3).alias("true")).sort("gb"))
print(x.filter(pl.col("rel") == "unrelated").group_by("rb").agg(pl.len(), pl.col("label").mean().round(3).alias("true")).sort("rb"))
# Parity: decoys "a few doors away" on the same side of the street would keep the parity.
x = x.with_columns(((pl.col("gap") % 2) == 0).alias("same_parity"))
print(x.filter(pl.col("rel") == "unrelated").group_by("same_parity").agg(pl.len(), pl.col("label").mean().round(3).alias("true")))
