"""Same-house decoys by name-change class. Training: per class, decoys at the Source 1 house vs at other houses of
the same street (placement ratio). France: other-house pairs of a class are nearly all decoys, so same-house decoys
~ other-house count x training ratio; estimated true rate of France same-house pairs per class, and how many file 11
accepts. Classes with estimated true rate below ~0.77 are drop candidates."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
classes = [("acronym", "g_acronym"), ("domain", "g_domain"), ("alias", "g_alias"), ("brand1", "g_brand1"), ("swap_common", "g_swap_common"),
           ("swap_filler", "g_swap_filler"), ("swap_rare", "g_swap_rare"), ("typo1", "g_typo1"), ("exact", "g_exact"), ("add", "g_add"),
           ("drop", "g_drop"), ("multi", "g_multi")]
lab = pl.lit("other")
for name, col in reversed(classes):
    lab = pl.when(pl.col(col) == 1).then(pl.lit(name)).otherwise(lab)


def frame(split, keep=None):
    g = pl.read_parquet(f"cache/gen/{split}.parquet")
    if keep is not None:
        g = g.join(keep, on="si", how="semi")
    cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos"]
    f = pl.concat([pl.read_parquet(p, columns=cols).join(g.select("si", "ti"), on=["si", "ti"], how="semi") for p in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))])
    g = g.join(f, on=["si", "ti"]).filter(pl.col("address_idf_cos") >= 0.6)
    g = g.with_columns(lab.alias("cls"), pl.when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1)
                       .then(pl.lit("other")).otherwise(pl.lit("unk")).alias("h"))
    return g.select("si", "ti", "cls", "h")


tr = frame("train")
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
tr = tr.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
T = tr.group_by("cls").agg(((pl.col("h") == "same") & (pl.col("label") == 0)).sum().alias("tr_same_dec"),
                           ((pl.col("h") == "other") & (pl.col("label") == 0)).sum().alias("tr_other_dec"),
                           ((pl.col("h") == "same")).sum().alias("tr_same"), ((pl.col("h") == "same") & (pl.col("label") == 1)).sum().alias("tr_same_true"))
T = T.with_columns((pl.col("tr_same_dec") / pl.col("tr_other_dec").clip(1)).alias("ratio"), (pl.col("tr_same_true") / pl.col("tr_same").clip(1)).alias("tr_same_rate"))

s1 = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
frs = s1.filter(pl.col("country") == "France").select("si")
fr = frame("test", frs)
links = (R("output/final_pre1_frcu/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("acc")))
fr = fr.join(links, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))
F = fr.group_by("cls").agg((pl.col("h") == "same").sum().alias("fr_same"), ((pl.col("h") == "same") & (pl.col("acc") == 1)).sum().alias("fr_same_acc"),
                           (pl.col("h") == "other").sum().alias("fr_other"))
out = T.join(F, on="cls").with_columns((pl.col("fr_other") * pl.col("ratio")).alias("fr_same_dec_est"))
out = out.with_columns((1 - pl.col("fr_same_dec_est") / pl.col("fr_same").clip(1)).clip(0, 1).alias("fr_same_true_est"))
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(200); pl.Config.set_float_precision(3)
print(out.select("cls", "tr_same", "tr_same_rate", "ratio", "fr_same", "fr_same_acc", "fr_other", "fr_same_dec_est", "fr_same_true_est").sort("cls"))
