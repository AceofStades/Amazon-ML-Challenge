"""Estimate the true share of France same-address common-word swaps from decoy placement in training:
decoy swaps land on the Source 1 house 0.42 times as often as on another house of the same street."""
from collections import Counter
from difflib import SequenceMatcher

import polars as pl

R = lambda p, c: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s = R("cache/test_source1.tsv", ["name_core"]).with_columns(R("dataset/test/test_source1.tsv", ["country"])["country"])
df = Counter()
for core in s.filter(pl.col("country") == "France")["name_core"]:
    df.update(set(core.split()))
x = pl.read_parquet("/tmp/fr_swaps.parquet")


def subsequence(short, long_):
    it = iter(long_)
    return all(ch in it for ch in short)


def common(a, b):
    if b in {"center", "services", "service", "partners", "france"} or df.get(b, 0) < 20 or SequenceMatcher(None, a, b).ratio() >= 0.75:
        return False
    return not (subsequence(b, a) or subsequence(a, b))


x = x.with_columns(pl.struct("miss", "extra").map_elements(lambda r: common(r["miss"], r["extra"]), return_dtype=pl.Boolean).alias("common"))
for flag in (True, False):
    c = x.filter(pl.col("common") == flag)
    eq = c.filter(pl.col("H") == "eq").height
    cf = c.filter(pl.col("H") == "cf").height
    hi = c.filter((pl.col("H") == "eq") & (pl.col("p1") >= 0.5)).height
    print(f"common={flag}: same house {eq:,}, other house same street {cf:,}; estimated true share at same house "
          f"{max(0.0, 1 - 0.42 * cf / max(eq, 1)):.2f}; same-house with p1 >= 0.5: {hi:,}")
top = x.filter(pl.col("common")).group_by("extra", "H").len().pivot(on="H", index="extra", values="len").fill_null(0)
top = top.with_columns((1 - 0.42 * pl.col("cf") / pl.col("eq").clip(1)).clip(0, 1).round(2).alias("true_share")).sort("eq", descending=True)
print(top.head(15).rows())
