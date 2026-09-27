import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
u = pl.read_parquet("cache/fr_union.parquet").filter((pl.col("h") == "other") & (pl.col("state") == "free"))
u = u.with_columns(pl.when(pl.col("g_swap_common") == 1).then(pl.lit("swap_common")).when(pl.col("g_brand1") == 1).then(pl.lit("brand1"))
                   .when(pl.col("g_acronym") == 1).then(pl.lit("acronym")).when(pl.col("g_domain") == 1).then(pl.lit("domain"))
                   .when(pl.col("g_typo1") == 1).then(pl.lit("typo1")).when(pl.col("g_alias") == 1).then(pl.lit("alias")).otherwise(pl.lit("rest")).alias("k"))
print(u.group_by("k").len().sort("len", descending=True).rows())
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
feat = pl.read_parquet("cache/class_audit_fr_p2.parquet", columns=["si", "ti"]).head(0)
for k in ("rest", "swap_common", "domain", "typo1"):
    s = u.filter(pl.col("k") == k).sample(min(7, u.filter(pl.col("k") == k).height), seed=2).join(
        s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
        tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
    print("---", k)
    for r in s.iter_rows(named=True):
        print(f"  p1={r['p1']:.2f} p2={r['p2']:.3f} | {r['sn'][:28]:28} {r['sa'][:34]:34} || {r['tn'][:28]:28} {r['ta'][:34]}")
