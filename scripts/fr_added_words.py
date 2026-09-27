"""France pairs: words the target name adds to the Source 1 name (raw, lowercased, accents folded, legal forms kept),
tabulated by house relation: accepted same-house pairs versus small-gap and large-gap conflicts. Decoys 'a few doors
away' reveal which added words are decoy signatures; the same words at the Source 1 house are then suspect."""
import glob
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def toks(s):
    return re.findall(r"[a-z0-9]+", fold(s))


def added(sn, tn):
    a = set(toks(sn))
    return " ".join(sorted(t for t in set(toks(tn)) if t not in a))


s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
x = pl.concat([pl.read_parquet(f, columns=cols).join(fr, on="si", how="semi").filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.8))
               for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
gapt = pl.read_parquet("cache/conflict_gap_test.parquet").filter(pl.col("country") == "france").select("si", "ti", "gap")
x = x.join(gapt, on=["si", "ti"], how="left")
sub = (R("output/final_pre1/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
       .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("acc")))
x = x.join(sub, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
x = x.join(s1.select("si", pl.col("business_name").alias("sn")), on="si").join(tg.select("ti", pl.col("business_name").alias("tn")), on="ti")
x = x.with_columns(pl.struct("sn", "tn").map_elements(lambda r: added(r["sn"], r["tn"]), return_dtype=pl.String).alias("add"))
x = x.with_columns(pl.when(pl.col("house_match") == 1).then(pl.lit("same")).when((pl.col("house_conflict") == 1) & (pl.col("gap") <= 50) & (pl.col("gap") >= 1))
                   .then(pl.lit("near")).when((pl.col("house_conflict") == 1) & (pl.col("gap") > 50)).then(pl.lit("far")).otherwise(pl.lit("other")).alias("where"))
x.select("si", "ti", "where", "gap", "acc", "add").write_parquet("cache/fr_added_words.parquet")
t = x.filter(pl.col("add") != "").group_by("add", "where").agg(pl.len().alias("n"), pl.col("acc").sum().alias("acc"))
w = t.pivot(on="where", index="add", values=["n", "acc"]).fill_null(0)
w = w.with_columns((pl.col("n_near") + pl.col("n_same")).alias("tot")).sort("tot", descending=True)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(200)
print(w.select("add", "n_same", "acc_same", "n_near", "acc_near", "n_far", "acc_far").head(35))
print("pairs with no added word:", x.filter(pl.col("add") == "").group_by("where").agg(pl.len(), pl.col("acc").sum()).sort("where"))
