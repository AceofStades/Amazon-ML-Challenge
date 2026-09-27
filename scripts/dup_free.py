"""For same-street similar-name pairs whose target's normalized name and address repeat in Sources 2/3: is the
target held by another entity in the submission, or free? Per class, France and US/India test; training truth shows
how often such a target belongs to the paired entity when no other entity holds it (oracle 'held' = true owner).
Usage: python scripts/dup_free.py SUBMISSION"""
import sys

import polars as pl

sys.argv, argv = sys.argv[:1], sys.argv
exec(open("scripts/dup_group_audit.py").read().split("pl.Config.set_tbl_rows")[0])
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(200); pl.Config.set_float_precision(4)
sub = argv[1]
for name, cs in (("france", ["france"]), ("us/india", ["us", "india"])):
    l = links("test", sub)
    x = pairs("test", cs).filter(pl.col("n_dup") >= 2).join(l, on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
    held = l.select("ti").unique().with_columns(pl.lit(1).alias("t_held"))
    x = x.join(held, on="ti", how="left").with_columns(pl.col("t_held").fill_null(0))
    x = x.with_columns(pl.when(pl.col("y") == 1).then(pl.lit("accepted")).when(pl.col("t_held") == 1).then(pl.lit("held_by_other"))
                       .otherwise(pl.lit("free")).alias("state"))
    print(name)
    print(x.group_by("h", "nd", "state").len().sort("h", "nd", "state").pivot(on="state", index=["h", "nd"], values="len").fill_null(0).sort("h", "nd"))
