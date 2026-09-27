"""Same-name, same-street, different-house candidates per Source 1 entity: training (with true rate) against test by
country (with the share the current submission accepts). If France had the training rate of true records with a
randomized house number, its accepted count would be far below the true count."""
import glob
import sys

import polars as pl

sys.path.insert(0, "scripts")
R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
exec(open("scripts/conflict_truth.py").read().split("def relation")[1].join(["def relation", ""]) if False else "")
src = open("scripts/conflict_truth.py").read()
rel_src = "def relation" + src.split("def relation")[1].split("\n\n\nx = x.with_columns")[0]
exec(rel_src)
cols = ["si", "ti", "house_conflict", "address_idf_cos", "name_idf_cos", "name_exact"]
x = pl.concat([pl.read_parquet(f, columns=cols).filter((pl.col("house_conflict") == 1) & (pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.95))
               for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
s1 = R("cache/test_source1.tsv", ["country", "house"]).with_row_index("si")
addr = pl.concat([R(f"cache/test_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
x = x.join(s1, on="si").with_columns(addr.gather(x["ti"].to_numpy()).alias("ta"))
x = x.with_columns(pl.struct("house", "ta").map_elements(lambda s: relation(s["house"], s["ta"]), return_dtype=pl.String).alias("rel"))
p1 = pl.concat([pl.read_parquet(f).join(x.select("si", "ti"), on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
x = x.join(p1, on=["si", "ti"], how="left")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
s1id = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
sub = (R(sys.argv[1]).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
       .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("acc")))
x = x.join(sub, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
x.select("si", "ti", "country", "rel", "name_exact", "p1", "acc").write_parquet("cache/conflict_test.parquet")
E = s1.group_by("country").len().rename({"len": "E"})
t = x.group_by("country", "rel", "name_exact").agg(pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc_share"),
                                                   (pl.col("p1") >= 0.9).mean().round(3).alias("p1_ge_.9")).join(E, on="country")
t = t.with_columns((pl.col("n") / pl.col("E")).round(4).alias("per_E")).drop("E")
tr = pl.read_parquet("cache/conflict_truth_train.parquet").group_by("rel", "name_exact").agg(
    (pl.len() / 2206821).round(4).alias("train_per_E"), pl.col("label").mean().round(3).alias("train_true"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(200)
print(t.filter(pl.col("name_exact") == 1).join(tr, on=["rel", "name_exact"], how="left").sort("rel", "country"))
