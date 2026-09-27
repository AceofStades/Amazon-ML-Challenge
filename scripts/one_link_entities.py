"""Entities with exactly one predicted link, by country: the class of that link (address-less, house relation, name
difference) against all links. A class over-represented among France one-link entities could be singleton false merges.
Usage: python scripts/one_link_entities.py SUBMISSION"""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s):
    return frozenset(re.findall(r"[a-z0-9]+", fold(s)))


sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_name", "business_address"]) for i in (2, 3)]).with_row_index("ti")
col = "matched_entity_ids"
m = R(sub).join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
m = m.with_columns(pl.when(pl.col(col) == "").then(0).otherwise(pl.col(col).str.count_matches(",") + 1).alias("n"))
l = (m.filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti", "n"))
feats = pl.concat([pl.read_parquet(f, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos"]).join(l, on=["si", "ti"], how="semi")
                   for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
l = l.join(feats, on=["si", "ti"], how="left")
l = l.with_columns(s1["country"].gather(l["si"].to_numpy()).alias("country"), s1["business_name"].gather(l["si"].to_numpy()).alias("sn"),
                   tg["business_name"].gather(l["ti"].to_numpy()).alias("tn"), (tg["business_address"].gather(l["ti"].to_numpy()) == "").alias("noaddr"))
nd = [("same" if tokens(a) == tokens(b) else "diff") for a, b in zip(l["sn"].to_list(), l["tn"].to_list())]
l = l.with_columns(pl.Series("nd", nd))
cls = (pl.when(pl.col("noaddr")).then(pl.lit("noaddr"))
       .when(pl.col("address_idf_cos") < 0.6).then(pl.lit("otherstreet"))
       .when(pl.col("house_match") == 1).then(pl.lit("samehouse"))
       .when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")))
l = l.with_columns(pl.concat_str([cls, pl.col("nd")], separator="/").alias("cls"), (pl.col("n") == 1).alias("one"))
t = l.group_by("country", "one", "cls").len()
tot = l.group_by("country", "one").len().rename({"len": "tot"})
t = t.join(tot, on=["country", "one"]).with_columns((pl.col("len") / pl.col("tot")).round(4).alias("share"))
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
print(t.pivot(on=["country", "one"], index="cls", values="share").sort("cls"))
