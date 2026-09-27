"""Are rejected same-house candidates of a class true records? If a class is true noise, an entity holding such a
rejected candidate lost one of its records, so its accepted link count is lower than size-matched peers; if the class is
decoys (extra records), its count is normal. Classes: pure made-up brand (single word >= 6 letters, not in the entity
name, not a handle/domain), common-word swap. Training rows use the truth (links excluding the class) as calibration.
Usage: python scripts/rejected_class_test.py SUBMISSION"""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def is_brand(sn, tn):
    t = re.findall(r"[A-Za-z]+", fold(tn))
    if len(t) != 1 or len(t[0]) < 6 or re.search(r"[#@.]|com$", tn or ""):
        return False
    w = t[0]
    s = re.findall(r"[a-z]+", fold(sn))
    return w not in s and not any(x and x in w for x in s if len(x) >= 4)


def run(split, path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "business_name", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("y")))
    ents = pl.read_parquet(f"cache/gen/{split}.parquet", columns=["si"]).unique() if split == "train" else s1.select("si")
    c = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "address_idf_cos"]).join(ents, on="si", how="semi")
                   .filter((pl.col("house_match") == 1) & (pl.col("address_idf_cos") >= 0.6)).select("si", "ti")
                   for p in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))])
    c = c.with_columns(s1["business_name"].gather(c["si"].to_numpy()).alias("sn"), tg["business_name"].gather(c["ti"].to_numpy()).alias("tn"))
    c = c.filter(pl.struct("sn", "tn").map_elements(lambda r: is_brand(r["sn"], r["tn"]), return_dtype=pl.Boolean))
    c = c.join(l, on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
    ll = l.join(ents, on="si", how="semi")
    n_all = ents.join(ll.group_by("si").len().rename({"len": "n"}), on="si", how="left").with_columns(pl.col("n").fill_null(0))
    # links excluding brand candidates (so accepted or true brands do not count)
    n_ex = ents.join(ll.join(c.select("si", "ti"), on=["si", "ti"], how="anti").group_by("si").len().rename({"len": "n_ex"}), on="si", how="left").with_columns(pl.col("n_ex").fill_null(0))
    flag = c.group_by("si").agg(pl.len().alias("n_brand_cand"), pl.col("y").sum().alias("n_brand_linked"))
    x = n_all.join(n_ex, on="si").join(flag, on="si", how="left").with_columns(pl.col("n_brand_cand").fill_null(0), pl.col("n_brand_linked").fill_null(0))
    x = x.join(s1.select("si", "country"), on="si")
    return x.with_columns((pl.col("n_brand_cand") > 0).alias("has_brand"), (pl.col("n_brand_linked") > 0).alias("brand_linked"))


pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
for split, path in (("train", "dataset/train/train_ground_truth.tsv"), ("test", sys.argv[1])):
    x = run(split, path)
    print(split)
    print(x.group_by("country", "has_brand", "brand_linked").agg(pl.len().alias("ents"), pl.col("n_ex").mean().round(3).alias("links_excl_brand"),
                                                                  pl.col("n_brand_cand").mean().round(2).alias("brand_cands"))
          .sort("country", "has_brand", "brand_linked"))
