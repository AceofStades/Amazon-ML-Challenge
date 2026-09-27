"""Label-free decoy test per link class. True noise replaces one of an entity's usual records, so entities holding a
link of that class have more links only through size bias (about +0.7 in training for every true class); a decoy
class adds a link on top of the entity's records (about +1.0). Classes: gen-feature kinds, house relation x name
difference, address-less, and how the link was added. Prints training truth (CV entities) and test submission.
Usage: python scripts/extra_link_test.py SUBMISSION"""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    return frozenset(re.findall(r"[a-z0-9]+", fold(s)))


def run(split, path, label):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_name", "business_address"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti"))
    if split == "train":   # restrict to the cross-validated entities (the gen cache covers them)
        ents = pl.read_parquet("cache/gen/train.parquet", columns=["si"]).unique()
        l = l.join(ents, on="si", how="semi")
    f = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]).join(l, on=["si", "ti"], how="semi")
                   for p in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))])
    l = l.join(f, on=["si", "ti"], how="left")
    sn = s1["business_name"].gather(l["si"].to_numpy()).to_list()
    tn = tg["business_name"].gather(l["ti"].to_numpy()).to_list()
    nd = []
    for a, b in zip(sn, tn):
        A, B = toks(a), toks(b)
        nd.append(("add" if B - A else "") + ("drop" if A - B else "") or "same")
    l = l.with_columns(pl.Series("nd", nd), (tg["business_address"].gather(l["ti"].to_numpy()) == "").alias("noaddr"),
                       s1["country"].gather(l["si"].to_numpy()).alias("country"))
    house = (pl.when(pl.col("noaddr")).then(pl.lit("noaddr"))
             .when(pl.col("address_idf_cos").is_null()).then(pl.lit("nofeat"))
             .when(pl.col("address_idf_cos") < 0.6).then(pl.lit("otherst"))
             .when(pl.col("house_match") == 1).then(pl.lit("same"))
             .when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")))
    l = l.with_columns(pl.concat_str([house, pl.col("nd")], separator="/").alias("cls"))
    l = l.with_columns(pl.when(pl.col("cls").str.starts_with("noaddr")).then(pl.lit("noaddr")).otherwise(pl.col("cls")).alias("cls"))
    n = l.group_by("si").len().rename({"len": "n"})
    has = l.select("si", "cls").unique().with_columns(pl.lit(True).alias("has"))
    out = []
    for c in l["cls"].unique().to_list():
        x = n.join(has.filter(pl.col("cls") == c).select("si", "has"), on="si", how="left").with_columns(pl.col("has").fill_null(False))
        x = x.join(s1.select("si", "country"), on="si")
        t = x.group_by("country", "has").agg(pl.len().alias("ents"), pl.col("n").mean().alias("m"))
        t = t.pivot(on="has", index="country", values=["ents", "m"]).with_columns(pl.lit(c).alias("cls"))
        out.append(t)
    t = pl.concat(out, how="diagonal").with_columns((pl.col("m_true") - pl.col("m_false")).round(3).alias("diff"), pl.lit(label).alias("src"))
    return t.select("src", "country", "cls", pl.col("ents_true").alias("ents_with"), "diff")


pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
a = run("train", "dataset/train/train_ground_truth.tsv", "train_truth")
b = run("test", sys.argv[1], "test_sub")
w = pl.concat([a, b]).filter(pl.col("ents_with") >= 200)
print(w.pivot(on=["src", "country"], index="cls", values="diff").sort("cls"))
print(w.pivot(on=["src", "country"], index="cls", values="ents_with").sort("cls"))
