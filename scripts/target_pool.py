"""Per-entity accounting of every Source 2/3 record: linked (training truth) or assigned (test submission), and for the
rest the class of the record's best stage-1 candidate (address-less; same or other street; similar or different name;
house relation; no candidate). Shows whether France's unassigned pool looks like US/India's decoy pool.
Usage: python scripts/target_pool.py SUBMISSION"""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def best_candidates(split):
    """Best stage-1 candidate of every target, computed part by part to bound memory."""
    cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
    s = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))])
    best = []
    for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet")):
        x = pl.read_parquet(f, columns=cols)
        x = x.join(s, on=["si", "ti"], how="left").with_columns(pl.col("p1").fill_null(0.0))
        best.append(x.sort("p1", descending=True).unique("ti", keep="first"))
    del s
    return pl.concat(best).sort("p1", descending=True).unique("ti", keep="first")


def classify(b):
    street = pl.when(pl.col("address_idf_cos") >= 0.6).then(pl.lit("street")).otherwise(pl.lit("otherst"))
    name = pl.when(pl.col("name_idf_cos") >= 0.8).then(pl.lit("sim")).otherwise(pl.lit("diff"))
    house = pl.when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conf")).otherwise(pl.lit("unk"))
    pb = pl.when(pl.col("p1") >= 0.5).then(pl.lit("hi")).when(pl.col("p1") >= 0.05).then(pl.lit("mid")).otherwise(pl.lit("lo"))
    return (pl.when(pl.col("noaddr")).then(pl.concat_str([pl.lit("noaddr"), pb], separator="/"))
            .when(pl.col("address_idf_cos").is_null()).then(pl.lit("nocand"))
            .otherwise(pl.concat_str([street, name, house, pb], separator="/")))


def pool(split, links_path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_address", "country"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(links_path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("ti").unique().with_columns(pl.lit(1).alias("linked")))
    b = best_candidates(split).select("ti", "p1", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos")
    t = tg.select("ti", "country", (pl.col("business_address") == "").alias("noaddr")).join(l, on="ti", how="left").join(b, on="ti", how="left")
    t = t.with_columns(pl.col("linked").fill_null(0), classify(t).alias("cls") if False else pl.lit(None))
    t = t.with_columns(classify(t).alias("cls")).with_columns(pl.when(pl.col("linked") == 1).then(pl.lit("LINKED")).otherwise(pl.col("cls")).alias("cls"))
    E = s1.group_by("country").len().rename({"len": "E"})
    return t.group_by("country", "cls").len().join(E, on="country").with_columns((pl.col("len") / pl.col("E")).round(4).alias("perE"))


tr = pool("train", "dataset/train/train_ground_truth.tsv").with_columns(pl.concat_str([pl.lit("train_"), pl.col("country")]).alias("col"))
te = pool("test", sys.argv[1]).with_columns(pl.concat_str([pl.lit("test_"), pl.col("country")]).alias("col"))
w = pl.concat([tr, te]).pivot(on="col", index="cls", values="perE").fill_null(0).sort("cls")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(8); pl.Config.set_tbl_width_chars(200)
print(w)
print(w.select(pl.all().exclude("cls").sum()))
w.write_parquet("cache/target_pool.parquet")
