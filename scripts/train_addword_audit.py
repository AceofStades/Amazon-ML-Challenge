"""Training truth for same-street pairs with similar names by house relation and name difference, broken down by the
added and dropped words (the English analogues of the France 'drop a type word, add Groupe/& Fils' pairs).
Usage: python scripts/train_addword_audit.py H ND [ROWS]"""
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


H, ND = sys.argv[1], sys.argv[2]
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 40
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
parts = []
for f in sorted(glob.glob("cache/v3/train/part*.parquet")):
    p = pl.read_parquet(f, columns=cols).filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.8))
    if H == "same":
        p = p.filter(pl.col("house_match") == 1)
    elif H == "unknown":
        p = p.filter((pl.col("house_match") != 1) & (pl.col("house_conflict") != 1))
    else:
        p = p.filter(pl.col("house_conflict") == 1)
    parts.append(p.select("si", "ti"))
x = pl.concat(parts)
raw1 = R("dataset/train/train_source1.tsv", ["business_name"])["business_name"]
raw2 = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
x = x.with_columns(raw1.gather(x["si"].to_numpy()).alias("sn"), raw2.gather(x["ti"].to_numpy()).alias("tn"))
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
x = x.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
out = []
for sn, tn, lab in x.select("sn", "tn", "label").iter_rows():
    a, b = tokens(sn), tokens(tn)
    d = ("add" if b - a else "") + ("drop" if a - b else "") or "same"
    if d == ND:
        out.append((" ".join(sorted(b - a)), " ".join(sorted(a - b)), lab, sn, tn))
d = pl.DataFrame(out, schema=["added", "dropped", "label", "sn", "tn"], orient="row")
pl.Config.set_tbl_rows(rows + 5); pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(200)
print(f"train {H}/{ND}: {d.height:,} pairs, true {d['label'].mean():.4f}")
for col in ("added", "dropped"):
    print(d.group_by(col).agg(pl.len().alias("n"), pl.col("label").mean().alias("true")).sort("n", descending=True).head(rows))
d.write_parquet(f"cache/train_{H}_{ND}_words.parquet")
