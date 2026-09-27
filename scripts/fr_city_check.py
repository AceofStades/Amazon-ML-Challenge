"""France: do accepted links agree on the city? French street names repeat across cities ("Rue de la Gare"), so a
same-name, same-number, same-street-name pair can still be two different places. Cities are the frequent
digit-free address fields of France Source 1 records (regions excluded). Reports accepted and unassigned pairs by
city agreement (same, different, target without a city).
Usage: python scripts/fr_city_check.py SUBMISSION"""
import glob
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return re.sub(r"[^a-z0-9 ]+", " ", "".join(ch for ch in s if not unicodedata.combining(ch)).lower()).strip()


s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France")
cnt = Counter()
for a in fr["business_address"].to_list():
    for f in a.split(","):
        f = fold(f)
        if f and not re.search(r"\d", f):
            cnt[f] += 1
regions = {"hauts de france", "nouvelle aquitaine", "pays de la loire", "auvergne rhone alpes", "ile de france", "bretagne",
           "occitanie", "grand est", "normandie", "provence alpes cote d azur", "bourgogne franche comte", "centre val de loire"}
cities = [c for c, n in cnt.most_common(400) if n >= 200 and c not in regions]
print(len(cities), "cities:", cities[:60])
pat = re.compile(r"\b(" + "|".join(sorted((re.escape(c) for c in cities), key=len, reverse=True)) + r")\b")


def city(a):
    m = pat.findall(fold(a))
    return m[-1] if m else ""


sub = sys.argv[1]
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(fr.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti"))
l = l.with_columns(s1["business_address"].gather(l["si"].to_numpy()).alias("sa"), tg["business_address"].gather(l["ti"].to_numpy()).alias("ta"))
cs = [city(a) for a in l["sa"].to_list()]
ct = [city(a) for a in l["ta"].to_list()]
l = l.with_columns(pl.Series("cs", cs), pl.Series("ct", ct))
l = l.with_columns(pl.when(pl.col("ta") == "").then(pl.lit("noaddr")).when(pl.col("ct") == "").then(pl.lit("t_nocity"))
                   .when(pl.col("cs") == "").then(pl.lit("s_nocity")).when(pl.col("cs") == pl.col("ct")).then(pl.lit("same"))
                   .otherwise(pl.lit("different")).alias("city"))
print(l.group_by("city").len().sort("len", descending=True))
d = l.filter(pl.col("city") == "different")
print(d.group_by("cs", "ct").len().sort("len", descending=True).head(25))
for r in d.sample(min(15, d.height), seed=1).iter_rows(named=True):
    print(f"  {r['sa'][:60]:60} || {r['ta'][:60]}")
l.write_parquet("cache/fr_city_links.parquet")
