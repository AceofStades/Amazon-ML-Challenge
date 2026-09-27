"""Scan of rejected candidate classes: for each class (house relation x name difference on same-street similar-name
pairs, plus address-less), entities holding an unaccepted candidate of the class whose target no entity holds, versus
entities without such a candidate: mean accepted links. A class of true records shows a deficit (the entity lost a
record); decoys show none. Training rows compute the same with truth, where 'accepted' means linked.
Usage: python scripts/rejected_class_scan.py SUBMISSION"""
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


def gap(h, ta):
    try:
        hv = int(h)
    except (TypeError, ValueError):
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta or "") if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


def run(split, path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_name", "business_address"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti"))
    ents = pl.read_parquet(f"cache/gen/{split}.parquet", columns=["si"]).unique() if split == "train" else s1.select("si")
    held = l.select("ti").unique()
    c = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]).join(ents, on="si", how="semi")
                   .filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.8))
                   for p in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))])
    c = c.join(held, on="ti", how="anti")          # unaccepted and target free
    house = R(f"cache/{split}_source1.tsv", ["house"])["house"]
    c = c.with_columns(s1["business_name"].gather(c["si"].to_numpy()).alias("sn"), tg["business_name"].gather(c["ti"].to_numpy()).alias("tn"),
                       house.gather(c["si"].to_numpy()).alias("hs"), tg["business_address"].gather(c["ti"].to_numpy()).alias("ta"))
    nd = []
    for a, b in zip(c["sn"].to_list(), c["tn"].to_list()):
        A, B = toks(a), toks(b)
        nd.append(("add" if B - A else "") + ("drop" if A - B else "") or "same")
    c = c.with_columns(pl.Series("nd", nd), pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
    h = (pl.when(pl.col("house_match") == 1).then(pl.lit("same"))
         .when(pl.col("house_conflict") != 1).then(pl.lit("unknown"))
         .when(pl.col("gap") <= 30).then(pl.lit("near")).otherwise(pl.lit("far")))
    c = c.with_columns(pl.concat_str([h, pl.col("nd")], separator="/").alias("cls")).select("si", "cls").unique()
    n = ents.join(l.group_by("si").len().rename({"len": "n"}), on="si", how="left").with_columns(pl.col("n").fill_null(0)).join(s1.select("si", "country"), on="si")
    base = n.group_by("country").agg(pl.col("n").mean().alias("base"))
    out = []
    for k in sorted(c["cls"].unique().to_list()):
        f = n.join(c.filter(pl.col("cls") == k).select("si"), on="si", how="semi")
        t = f.group_by("country").agg(pl.len().alias("ents"), pl.col("n").mean().alias("m")).join(base, on="country")
        out.append(t.with_columns(pl.lit(k).alias("cls"), (pl.col("m") - pl.col("base")).round(3).alias("delta")))
    return pl.concat(out).select("cls", "country", "ents", "delta")


pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
tr = run("train", "dataset/train/train_ground_truth.tsv").with_columns(pl.lit("train").alias("split"))
te = run("test", sys.argv[1]).with_columns(pl.lit("test").alias("split"))
x = pl.concat([tr, te]).filter(pl.col("ents") >= 150)
print(x.pivot(on=["split", "country"], index="cls", values="delta").sort("cls"))
print(x.pivot(on=["split", "country"], index="cls", values="ents").sort("cls"))
