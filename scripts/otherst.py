"""Pairs whose street differs (address IDF cosine < 0.6) but whose name is similar (name IDF cosine >= 0.8):
training true rate by house relation and name-core uniqueness k (Source 1 entities of the country sharing the core);
France test counts, file 11 acceptance, 6-model blend p2."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def frame(split, country=None):
    s1 = R(f"cache/{split}_source1.tsv", ["country", "name_core"]).with_row_index("si")
    k = s1.group_by("country", "name_core").len().rename({"len": "k"})
    s1 = s1.join(k, on=["country", "name_core"])
    keep = s1.filter(pl.col("country") == country).select("si") if country else None
    cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos", "name_exact"]
    parts = []
    for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet")):
        p = pl.read_parquet(f, columns=cols).filter((pl.col("address_idf_cos") < 0.6) & (pl.col("name_idf_cos") >= 0.8))
        parts.append(p.join(keep, on="si", how="semi") if keep is not None else p)
    x = pl.concat(parts).join(s1.select("si", "k"), on="si")
    addr = pl.concat([R(f"cache/{split}_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
    x = x.with_columns(addr.gather(x["ti"].to_numpy()).alias("ta")).filter(pl.col("ta") != "")
    return x.with_columns(pl.when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict"))
                          .otherwise(pl.lit("unknown")).alias("h"),
                          pl.when(pl.col("k") == 1).then(pl.lit("k1")).when(pl.col("k") <= 3).then(pl.lit("k2-3")).otherwise(pl.lit("k4+")).alias("kb"))


tr = frame("train")
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
tr = tr.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
T = tr.group_by("h", "name_exact", "kb").agg((pl.len() / 2206821).round(4).alias("tr_perE"), pl.col("label").mean().round(3).alias("tr_true"))

fr = frame("test", "france")
s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
links = (R("output/final_safe_own/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
fr = fr.join(links.with_columns(pl.lit(1).alias("acc")), on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
fr = fr.join(links.select("ti").unique().with_columns(pl.lit(1).alias("held")), on="ti", how="left").with_columns(pl.col("held").fill_null(0))
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"])
fr = fr.join(sc, on=["si", "ti"], how="left")
fr.write_parquet("cache/fr_otherst.parquet")
F = fr.group_by("h", "name_exact", "kb").agg((pl.len() / 259452).round(4).alias("fr_perE"), pl.len().alias("fr_n"), pl.col("acc").mean().round(3).alias("fr_acc"),
                                             ((pl.col("held") == 0)).sum().alias("fr_free"), ((pl.col("held") == 0) & (pl.col("p2") > 0.5)).sum().alias("free_p2_gt_.5"))
T.write_parquet("cache/otherst_train.parquet"); F.write_parquet("cache/otherst_fr.parquet")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(12)
print(T.join(F, on=["h", "name_exact", "kb"], how="full", coalesce=True).sort("h", "name_exact", "kb").select("h", "name_exact", "kb", "tr_perE", "tr_true", "fr_perE", "fr_acc", "fr_free", "free_p2_gt_.5"))
