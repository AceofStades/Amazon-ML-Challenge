"""Short-code records (a single 2-4 letter token) by country: how many exist per Source 1 entity, and whether their
letters spell the initials of a Source 1 entity at the same normalized address. If random short-code decoys are common
(France), some spell a neighbour's initials by chance: the letter-pair distribution of the non-matching codes estimates
how many matches chance alone produces.
Usage: python scripts/acronym_chance.py SUBMISSION"""
import re
import sys
import unicodedata
from collections import Counter

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
STOP = {"de", "du", "des", "la", "le", "les", "et", "d", "l", "of", "the", "and", "a", "&"}
LF = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "llc", "inc", "corp", "ltd", "limited", "private", "pvt", "llp", "lp", "co"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def code(name):
    t = re.findall(r"[A-Za-z]+", fold(name))
    return t[0].upper() if len(t) == 1 and 2 <= len(t[0]) <= 4 and t[0].isupper() else None


def initials(name):
    t = [w for w in re.findall(r"[a-z]+", fold(name).lower()) if w not in STOP and w not in LF]
    return "".join(w[0] for w in t).upper()


sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
n1 = R("cache/test_source1.tsv", ["address", "house"])
n2 = pl.concat([R(f"cache/test_source{i}.tsv", ["address", "house"]) for i in (2, 3)])
tg = tg.with_columns(n2["address"].alias("addr"))
s1 = s1.with_columns(n1["address"].alias("addr"), s1["business_name"].map_elements(initials, return_dtype=pl.String).alias("ini"))
tg = tg.with_columns(tg["business_name"].map_elements(code, return_dtype=pl.String).alias("code")).filter(pl.col("code").is_not_null())
col = "matched_entity_ids"
acc = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col).select(pl.col(col).alias("entity_id"))
       .with_columns(pl.lit(1).alias("assigned")))
tg = tg.join(acc, on="entity_id", how="left").with_columns(pl.col("assigned").fill_null(0))
E = s1.group_by("country").len().rename({"len": "E"})
# Source 1 entities at the same normalized address.
same = tg.filter(pl.col("addr") != "").select("ti", "code", "addr", "country", "assigned").join(
    s1.select("si", "addr", "ini"), on="addr", how="inner")
same = same.with_columns((pl.col("ini") == pl.col("code")).alias("match"))
per_t = same.group_by("ti", "country", "code", "assigned").agg(pl.col("match").any().alias("match"), pl.len().alias("n_s1_at_addr"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print("short-code records per entity:", tg.group_by("country").len().join(E, on="country").with_columns((pl.col("len") / pl.col("E")).round(4).alias("perE")).rows())
print(per_t.group_by("country", "match", "assigned").agg(pl.len().alias("n"), pl.col("n_s1_at_addr").mean().round(2).alias("s1_at_addr")).sort("country", "match", "assigned"))
# Chance estimate for France: P(code == initials of a given entity) from the non-matching codes' distribution.
for c in ("France", "US", "India"):
    nm = per_t.filter((pl.col("country") == c) & ~pl.col("match"))
    dist = Counter(nm["code"].to_list())
    tot = sum(dist.values())
    if not tot:
        continue
    at = same.filter((pl.col("country") == c)).join(nm.select("ti"), on="ti", how="semi")
    # expected chance matches = sum over (code record, entity at its address) of P(code) where the entity's initials == code
    # approximated by: for each non-matching record, the probability that a random code from the distribution equals
    # the initials of an entity at its address.
    ini_at = at.group_by("ti").agg(pl.col("ini"))
    exp = sum(sum(dist.get(i, 0) / tot for i in set(inis)) for inis in ini_at["ini"].to_list())
    m = per_t.filter((pl.col("country") == c) & pl.col("match")).height
    print(f"{c}: codes at a Source 1 address {per_t.filter(pl.col('country') == c).height:,}; matching initials {m:,}; "
          f"chance matches expected if every code were a random decoy drawn like the non-matching ones: {exp:,.0f} "
          f"(per non-matching record {exp / max(nm.height, 1):.4f})")
