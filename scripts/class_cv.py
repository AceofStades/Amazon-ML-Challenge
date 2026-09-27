"""Training out-of-fold blend p2 (output/cv_pre1.parquet, 220k entities) by class (house relation x name difference):
true rate per p2 bin, to size how precise a blend-based France restore would be for each class."""
import sys

import polars as pl

sys.argv = [sys.argv[0], "unused"]
src = open("scripts/class_audit.py").read().split("\nsub = sys.argv[1]")[0]
exec(src)
cv = pl.read_parquet("output/cv_pre1.parquet", columns=["si", "ti", "label", "p2"])
tr = frame("train").join(cv, on=["si", "ti"], how="inner")
tr = tr.with_columns(pl.col("p2").cut([0.5, 0.9, 0.98]).alias("pb"))
t = tr.group_by("h", "nd", "pb").agg(pl.len().alias("n"), pl.col("label").mean().round(4).alias("true"))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
w = t.with_columns(pl.format("{} ({})", pl.col("true"), pl.col("n")).alias("cell")).pivot(on="pb", index=["h", "nd"], values="cell")
print(w.sort("h", "nd"))
