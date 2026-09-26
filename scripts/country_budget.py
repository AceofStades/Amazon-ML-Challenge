"""Per-country target budget: target records per Source 1 entity (with and without address), accepted per entity,
and unassigned targets, for test (a submission file) and training (ground truth)."""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def budget(split, links):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "country"])
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_address", "country"]) for i in (2, 3)])
    tg = tg.with_columns((pl.col("business_address") == "").alias("noaddr"))
    E = s1.group_by("country").len().rename({"len": "E"})
    T = tg.group_by("country").agg(pl.len().alias("T"), pl.col("noaddr").sum().alias("T_noaddr"))
    links = links.join(tg.rename({"entity_id": "t", "country": "t_country"}), on="t")
    links = links.join(s1.rename({"entity_id": "s"}), on="s")
    A = links.group_by("country").agg(pl.len().alias("A"), pl.col("noaddr").sum().alias("A_noaddr"),
                                      (pl.col("t_country") != pl.col("country")).sum().alias("A_xcountry"))
    assigned = links.select("t").unique()
    U = tg.join(assigned, left_on="entity_id", right_on="t", how="anti").group_by("country").agg(
        pl.len().alias("U"), pl.col("noaddr").sum().alias("U_noaddr"))
    out = E.join(T, on="country", how="left").join(A, on="country", how="left").join(U, on="country", how="left")
    out = out.with_columns([(pl.col(c) / pl.col("E")).round(4).alias(c + "/E") for c in ("T", "T_noaddr", "A", "A_noaddr", "U", "U_noaddr")])
    pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
    print(split, out.select("country", "E", "T/E", "T_noaddr/E", "A/E", "A_noaddr/E", "U/E", "U_noaddr/E", "A_xcountry").sort("country"))


def explode(path, key):
    return (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .select(pl.col(key).alias("s"), pl.col("matched_entity_ids").alias("t")))


budget("train", explode("dataset/train/train_ground_truth.tsv", "source1_entity_id"))
budget("test", explode(sys.argv[1], "source1_entity_id"))
