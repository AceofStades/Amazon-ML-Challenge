"""US/India test: same-street similar-name pairs by house relation and name difference, broken down by the added word,
with the share a submission accepts. Compares with the training truth from train_addword_audit.py.
Usage: python scripts/usin_addword_accept.py SUBMISSION H ND [ROWS]"""
import glob
import sys

import polars as pl

import re
import unicodedata

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s):
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
    return set(out)


argv = sys.argv

sub, H, ND = argv[1], argv[2], argv[3]
rows = int(argv[4]) if len(argv) > 4 else 30
s1 = R("dataset/test/test_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)]).with_row_index("ti")
keep = s1.filter(pl.col("country").is_in(["US", "India"])).select("si")
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
parts = []
for f in sorted(glob.glob("cache/v3/test/part*.parquet")):
    p = pl.read_parquet(f, columns=cols).join(keep, on="si", how="semi").filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.8))
    p = p.filter(pl.col("house_match") == 1) if H == "same" else p.filter(pl.col("house_conflict") == 1)
    parts.append(p.select("si", "ti"))
x = pl.concat(parts)
x = x.with_columns(s1["business_name"].gather(x["si"].to_numpy()).alias("sn"), tg["business_name"].gather(x["ti"].to_numpy()).alias("tn"),
                   s1["country"].gather(x["si"].to_numpy()).alias("country"))
links = (R(sub).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("acc")))
x = x.join(links, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
out = []
for sn, tn, acc, c, si, ti in x.select("sn", "tn", "acc", "country", "si", "ti").iter_rows():
    a, b = tokens(sn), tokens(tn)
    d = ("add" if b - a else "") + ("drop" if a - b else "") or "same"
    if d == ND:
        out.append((si, ti, c, " ".join(sorted(b - a)), " ".join(sorted(a - b)), acc))
d = pl.DataFrame(out, schema=["si", "ti", "country", "added", "dropped", "acc"], orient="row")
pl.Config.set_tbl_rows(rows + 5); pl.Config.set_tbl_width_chars(200)
print(f"test US/India {H}/{ND}: {d.height:,} pairs, accepted {d['acc'].mean():.4f}")
print(d.group_by("added").agg(pl.len().alias("n"), pl.col("acc").mean().alias("acc"), pl.col("acc").sum().alias("n_acc"))
      .sort("n", descending=True).head(rows))
d.write_parquet(f"cache/usin_{H}_{ND}_accept.parquet")
