"""Where an added company word sits relative to the legal form, by class and truth (training) or acceptance (test).
Training shows two processes: true filler noise appends the word after the legal form ("Nova LLC Services"), decoys
insert it before the legal form or replace it ("Gold Brothers Group LLP", "Solstice LLC" -> "Solstice Group").
Usage: python scripts/suffix_position.py SUBMISSION"""
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
LF = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "llc", "inc", "corp", "corporation", "ltd", "limited", "private", "pvt",
      "llp", "lp", "co", "company", "pc", "pllc", "plc", "incorporated"}
WORDS = {"groupe", "developpement", "fils", "associes", "cie", "services", "france", "compagnie", "holding", "participations",
         "international", "distribution", "group", "holdings", "enterprises", "industries", "ventures", "overseas", "exports",
         "infratech", "center", "partners", "service"}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    out, run = [], ""
    for w in re.findall(r"[a-z0-9]+", fold(s)):
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return out


def position(sn, tn):
    a, b = set(toks(sn)), toks(tn)
    added = [w for w in b if w not in a and w in WORDS]
    if len(added) != 1:
        return None, None
    w = added[0]
    i = b.index(w)
    lf = [j for j, t in enumerate(b) if t in LF]
    if not lf:
        return w, "no_lf"
    return w, ("after_lf" if i > max(lf) else "before_lf")


def frame(parquet, split):
    x = pl.read_parquet(parquet)
    raw1 = R(f"dataset/{split}/{split}_source1.tsv", ["business_name"])["business_name"]
    raw2 = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
    sn, tn = raw1.gather(x["si"].to_numpy()).to_list(), raw2.gather(x["ti"].to_numpy()).to_list()
    res = [position(a, b) for a, b in zip(sn, tn)]
    return x.with_columns(pl.Series("word", [r[0] for r in res]), pl.Series("pos", [r[1] for r in res])).filter(pl.col("word").is_not_null())


pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
fr = frame("cache/class_audit_fr.parquet", "test")
fr = fr.with_columns(pl.when(pl.col("h") == "same").then(pl.lit("same")).when(pl.col("h").is_in(["near", "gap0"])).then(pl.lit("near"))
                     .otherwise(pl.col("h")).alias("hh"))
t = fr.group_by("hh", "pos").agg(pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc")).sort("hh", "pos")
print("France pairs with one added company word, by house relation and position:")
print(t)
print(fr.filter(pl.col("hh") == "same").group_by("word", "pos").agg(pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc"))
      .sort("word", "pos"))
fr.write_parquet("cache/fr_suffix_position.parquet")
