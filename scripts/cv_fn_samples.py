import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
e = pl.read_parquet("cache/cv_errors.parquet").filter((pl.col("kind") == "FN") & ~pl.col("noaddr"))
full = pl.read_parquet("cache/compose_cv_extra.parquet").filter(pl.col("label") == 1).select("si", "ti", "p_new")
e = e.join(full, on="si").filter(pl.col("p_new") == pl.col("p")).unique(["si", "ti"])
s1 = R("dataset/train/train_source1.tsv").with_row_index("si").with_columns(pl.col("si").cast(pl.UInt32))
tg = pl.concat([R(f"dataset/train/train_source{i}.tsv") for i in (2, 3)]).with_row_index("ti").with_columns(pl.col("ti").cast(pl.UInt32))
gen = pl.read_parquet("cache/gen/train.parquet")
classes = [("acronym", "g_acronym"), ("domain", "g_domain"), ("alias", "g_alias"), ("brand1", "g_brand1"), ("swap_common", "g_swap_common"),
           ("swap_filler", "g_swap_filler"), ("swap_rare", "g_swap_rare"), ("typo1", "g_typo1"), ("exact", "g_exact"), ("add", "g_add"), ("drop", "g_drop"), ("multi", "g_multi")]
lab = pl.lit("other")
for name, col in reversed(classes):
    lab = pl.when(pl.col(col) == 1).then(pl.lit(name)).otherwise(lab)
e = e.join(gen, on=["si", "ti"], how="left").with_columns(lab.alias("cls"))
print(e.group_by("cls").agg(pl.len(), pl.col("p").median().round(3)).sort("len", descending=True).rows())
s = e.sample(20, seed=4).join(s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
    tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
for r in s.iter_rows(named=True):
    print(f"  {r['cls']:11} p={r['p']:.2f} | {r['sn'][:26]:26} {r['sa'][:30]:30} || {r['tn'][:26]:26} {r['ta'][:30]}")
