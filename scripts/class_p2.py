"""Blend p2 by class (house relation x name difference) for France against US test pairs of the same class: if the
France p2 distribution in a class looks like the US one, the blend's separation transfers to that class."""
import polars as pl

fr = pl.read_parquet("cache/class_audit_fr.parquet")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"])
fr = fr.join(sc, on=["si", "ti"], how="left")
fr.write_parquet("cache/class_audit_fr_p2.parquet")
bins = [0.1, 0.5, 0.9, 0.98]
t = fr.filter(pl.col("p2").is_not_null()).with_columns(pl.col("p2").cut(bins).alias("pb")).group_by("h", "nd", "pb").agg(
    pl.len().alias("n"), pl.col("acc").mean().round(3).alias("acc"))
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(200)
w = t.pivot(on="pb", index=["h", "nd"], values="n").fill_null(0)
print("France pairs by p2 bin (count):")
print(w.sort("h", "nd"))
print("France: pairs without a p2 score by class:", fr.filter(pl.col("p2").is_null()).group_by("h", "nd").len().sort("h", "nd").rows())
