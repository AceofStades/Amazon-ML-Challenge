"""Training: same-street house conflicts with a similar name, by gap range and by whether the raw target name adds
words to the Source 1 name. Do decoys 'a few doors away' always change the name, so that exact raw names near the
Source 1 house are true records with a moved number?"""
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def diff(sn, tn):
    a, b = set(re.findall(r"[a-z0-9]+", fold(sn))), set(re.findall(r"[a-z0-9]+", fold(tn)))
    return ("add" if b - a else "") + ("drop" if a - b else "") or "same"


x = pl.read_parquet("cache/conflict_truth_train.parquet")
s1 = R("cache/train_source1.tsv", ["name", "house"])
tg = pl.concat([R(f"cache/train_source{i}.tsv", ["name", "address"]) for i in (2, 3)])
raw1 = R("dataset/train/train_source1.tsv", ["business_name"])["business_name"]
raw2 = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
x = x.with_columns(raw1.gather(x["si"].to_numpy()).alias("sn"), raw2.gather(x["ti"].to_numpy()).alias("tn"),
                   s1["house"].fill_null("").gather(x["si"].to_numpy()).alias("hs"), tg["address"].fill_null("").gather(x["ti"].to_numpy()).alias("ta"))


def gap(h, ta):
    try:
        hv = int(h)
    except ValueError:
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta) if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


x = x.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"),
                   pl.struct("sn", "tn").map_elements(lambda s: diff(s["sn"], s["tn"]), return_dtype=pl.String).alias("nd"))
x = x.with_columns(pl.when(pl.col("gap") == 0).then(pl.lit("0")).when(pl.col("gap") <= 50).then(pl.lit("near")).otherwise(pl.lit("far")).alias("g"))
print(x.group_by("g", "nd").agg(pl.len().alias("n"), pl.col("label").mean().round(3).alias("true")).sort("g", "nd"))
