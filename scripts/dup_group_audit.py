"""Do decoys come in groups? For same-street similar-name pairs (by house relation and name difference), the true rate
split by how many Source 2/3 records share the target's normalized name and address (1 = the record is alone).
Training truth, then France and US/India test acceptance of the same split.
Usage: python scripts/dup_group_audit.py SUBMISSION"""
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
    return frozenset(out)


def pairs(split, countries=None):
    s1 = R(f"cache/{split}_source1.tsv", ["country"]).with_row_index("si")
    keep = s1.filter(pl.col("country").is_in(countries)).select("si") if countries else None
    cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
    parts = []
    for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet")):
        p = pl.read_parquet(f, columns=cols).filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.8))
        if keep is not None:
            p = p.join(keep, on="si", how="semi")
        parts.append(p.with_columns(pl.when(pl.col("house_match") == 1).then(pl.lit("same"))
                                    .when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")).alias("h"))
                     .select("si", "ti", "h"))
    x = pl.concat(parts)
    tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["name", "address"]) for i in (2, 3)]).with_row_index("ti")
    grp = tg.group_by("name", "address").agg(pl.len().alias("n_dup"))
    tg = tg.join(grp, on=["name", "address"], how="left")
    x = x.join(tg.select("ti", "n_dup"), on="ti", how="left")
    raw1 = R(f"dataset/{split}/{split}_source1.tsv", ["business_name"])["business_name"]
    raw2 = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
    sn, tn = raw1.gather(x["si"].to_numpy()).to_list(), raw2.gather(x["ti"].to_numpy()).to_list()
    nd = []
    for a, b in zip(sn, tn):
        A, B = tokens(a), tokens(b)
        nd.append(("add" if B - A else "") + ("drop" if A - B else "") or "same")
    return x.with_columns(pl.Series("nd", nd), pl.col("n_dup").clip(1, 3).alias("dup"))


def links(split, path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    return (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
            .join(s1, left_on="source1_entity_id", right_on="entity_id").join(tg, left_on=col, right_on="entity_id")
            .select("si", "ti").with_columns(pl.lit(1).alias("y")))


pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(200); pl.Config.set_float_precision(4)
tr = pairs("train").join(links("train", "dataset/train/train_ground_truth.tsv"), on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
t = tr.group_by("h", "nd", "dup").agg((pl.len() / 2206821).alias("tr_perE"), pl.col("y").mean().alias("tr_true"))
res = [t]
for name, cs, n in (("fr", ["france"], 259452), ("us", ["us", "india"], 1473092)):
    te = pairs("test", cs).join(links("test", sys.argv[1]), on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
    res.append(te.group_by("h", "nd", "dup").agg((pl.len() / n).alias(f"{name}_perE"), pl.col("y").mean().alias(f"{name}_acc")))
out = res[0].join(res[1], on=["h", "nd", "dup"], how="full", coalesce=True).join(res[2], on=["h", "nd", "dup"], how="full", coalesce=True)
print(out.sort("h", "nd", "dup"))
