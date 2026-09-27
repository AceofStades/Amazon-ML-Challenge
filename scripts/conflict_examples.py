"""Examples of unassigned France exact-name same-street conflicts that stage 1 scores high, by gap range."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
x = pl.read_parquet("cache/conflict_fr_unassigned_high.parquet")
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
x = x.join(s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
    tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
print("pairs", x.height, "entities", x["si"].n_unique(), "targets", x["ti"].n_unique())
for lo, hi in ((1, 2), (3, 50), (51, 100000)):
    s = x.filter((pl.col("gap") >= lo) & (pl.col("gap") <= hi)).sample(8, seed=7)
    print(f"--- gap {lo}-{hi}")
    for r in s.iter_rows(named=True):
        print(f"p1={r['p1']:.2f} gap={r['gap']} | {r['sn'][:26]:26} {r['sa'][:40]:40} || {r['tn'][:26]:26} {r['ta'][:40]}")
