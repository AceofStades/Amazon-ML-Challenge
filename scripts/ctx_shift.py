"""Train/test shift of the context features within comparable strata: address-less or not, stage-1 p1 band."""
import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
out = []
for split in ("train", "test"):
    c = pl.read_parquet(f"cache/ctx/{split}.parquet")
    p1 = pl.read_parquet(f"cache/ce/{split}_scores.parquet", columns=["si", "ti"]).with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
    import glob
    pp = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))]).join(p1, on=["si", "ti"], how="semi")
    addr = pl.concat([R(f"cache/{split}_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
    c = c.join(pp, on=["si", "ti"]).with_columns(addr.gather(c.join(pp, on=["si", "ti"])["ti"].to_numpy()).alias("ta") if False else pl.lit(None))
    na = pl.DataFrame({"ti": pl.Series(range(addr.len()), dtype=pl.UInt32), "noaddr": addr == ""})
    c = c.join(na, on="ti").with_columns(pl.col("p1").cut([0.1, 0.5, 0.9]).alias("band"), pl.lit(split).alias("split"))
    out.append(c.group_by("split", "noaddr", "band").agg(pl.len().alias("n"), pl.col("m1").mean().round(2).alias("m1"), pl.col("q1").mean().round(3).alias("q1"),
                                                         pl.col("s_tok_eq").mean().round(3).alias("tok"), pl.col("top_other1").mean().round(3).alias("top_o")))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(pl.concat(out).sort("noaddr", "band", "split"))
