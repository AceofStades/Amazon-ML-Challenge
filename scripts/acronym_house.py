"""Acronym-named targets (gen feature g_acronym with initials that spell the entity's words) by country and house
relation to the entity (same house, conflict gap bins, unknown, other street), with the accepted share. Training
truth for the same split shows how often acronym records sit a few doors away when they are decoys.
Usage: python scripts/acronym_house.py SUBMISSION"""
import glob
import re
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def gap(h, ta):
    try:
        hv = int(h)
    except (TypeError, ValueError):
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta or "") if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


def frame(split, path):
    g = pl.read_parquet(f"cache/gen/{split}.parquet", columns=["si", "ti", "g_acronym", "g_consistent"]).filter(pl.col("g_acronym") > 0.5)
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("y")))
    f = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos"]).join(g, on=["si", "ti"], how="semi")
                   for p in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))])
    g = g.join(f, on=["si", "ti"], how="left").join(l, on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
    g = g.join(s1.select("si", "country"), on="si")
    house = R(f"cache/{split}_source1.tsv", ["house"])["house"]
    ta = tg["business_address"]
    g = g.with_columns(house.gather(g["si"].to_numpy()).alias("hs"), ta.gather(g["ti"].to_numpy()).alias("ta"))
    g = g.with_columns(pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
    rel = (pl.when(pl.col("ta") == "").then(pl.lit("noaddr"))
           .when(pl.col("address_idf_cos").is_null() | (pl.col("address_idf_cos") < 0.6)).then(pl.lit("otherst"))
           .when(pl.col("house_match") == 1).then(pl.lit("same"))
           .when(pl.col("house_conflict") != 1).then(pl.lit("unknown"))
           .when(pl.col("gap") <= 30).then(pl.lit("near1-30"))
           .otherwise(pl.lit("far>30")))
    return g.with_columns(rel.alias("rel"), (pl.col("g_consistent") > 0.5).alias("cons"))


pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
tr = frame("train", "dataset/train/train_ground_truth.tsv")
print("training (CV entities only):")
print(tr.group_by("country", "cons", "rel").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("true")).sort("country", "cons", "rel"))
te = frame("test", sys.argv[1])
print("test:")
print(te.group_by("country", "cons", "rel").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("acc")).sort("country", "cons", "rel"))
te.filter(pl.col("country") == "France").write_parquet("cache/fr_acronym_rel.parquet")
