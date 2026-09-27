"""Accepted links to address-less targets: how many Source 1 entities of the country share the target's core name
(tokens without legal forms, fillers and stop words). k >= 2 means the owner is ambiguous and the link is a guess.
Compared across countries; France's owners are picked by stage 1 alone (no composition).
Usage: python scripts/noaddr_core_ambiguity.py SUBMISSION"""
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
DROP = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "llc", "inc", "corp", "corporation", "ltd", "limited", "private", "pvt",
        "llp", "lp", "co", "company", "pc", "pllc", "plc", "incorporated", "cie", "compagnie", "fils", "freres", "frs", "associes",
        "groupe", "developpement", "services", "service", "france", "india", "center", "centre", "partners", "the", "and", "et",
        "de", "du", "des", "la", "le", "les", "of", "d", "l", "s", "a"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def core(s):
    return " ".join(sorted(set(w for w in re.findall(r"[a-z0-9]+", fold(s)) if w not in DROP)))


sub = sys.argv[1]
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
s1 = s1.with_columns(pl.col("business_name").map_elements(core, return_dtype=pl.String).alias("core"))
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti", "business_name", "business_address"), left_on=col, right_on="entity_id")
     .filter(pl.col("business_address") == "").select("si", "ti", "country", "business_name"))
l = l.with_columns(pl.col("business_name").map_elements(core, return_dtype=pl.String).alias("tcore"))
k = s1.group_by("country", "core").len().rename({"len": "k", "core": "tcore"})
l = l.join(k, on=["country", "tcore"], how="left").with_columns(pl.col("k").fill_null(0))
l = l.join(s1.select("si", pl.col("core").alias("score")), on="si")
l = l.with_columns((pl.col("score") == pl.col("tcore")).alias("own_core_eq"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(l.with_columns(pl.col("k").clip(0, 4)).group_by("country", "own_core_eq", "k").len().sort("country", "own_core_eq", "k")
      .pivot(on="country", index=["own_core_eq", "k"], values="len"))
# unassigned address-less targets for comparison
un = tg.filter(pl.col("business_address") == "").join(l.select("ti"), on="ti", how="anti").select("ti", "country", "business_name")
un = un.with_columns(pl.col("business_name").map_elements(core, return_dtype=pl.String).alias("tcore"), pl.col("country"))
un = un.join(k, on=["country", "tcore"], how="left").with_columns(pl.col("k").fill_null(0).clip(0, 4))
print("unassigned address-less targets by k:")
print(un.group_by("country", "k").len().sort("country", "k").pivot(on="country", index="k", values="len"))
