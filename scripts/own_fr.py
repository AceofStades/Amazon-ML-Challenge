import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
o = pl.read_parquet("cache/noaddr_owner_test.parquet").filter((pl.col("country") == "France") & (pl.col("pm") >= 0.9))
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p2"])
o = o.join(sc, on=["si", "ti"], how="left")
print("France owner picks:", o.height, "p2 > 0.5:", (o["p2"] > 0.5).sum(), "p2 < 0.05:", (o["p2"] < 0.05).sum(), "q median:", round(o["q"].median(), 3))
s = o.sample(12, seed=3).join(s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
    tg.select("ti", pl.col("business_name").alias("tn")), on="ti")
for r in s.iter_rows(named=True):
    print(f"  pm={r['pm']:.2f} q={r['q']:.2f} p1={r['p1']:.2f} p2={r['p2']:.2f} | {r['sn'][:30]:30} {r['sa'][:30]:30} || {r['tn'][:34]}")
