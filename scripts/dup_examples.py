"""Examples of same-street similar-name pairs whose target's normalized name and address repeat in Sources 2/3.
Usage: python scripts/dup_examples.py SPLIT COUNTRY H ND [N]   (H: same/conflict/unknown; prints label on train)"""
import sys

import polars as pl

sys.argv, argv = sys.argv[:1], sys.argv
exec(open("scripts/dup_group_audit.py").read().split("pl.Config.set_tbl_rows")[0])
split, country, H, ND = argv[1], argv[2], argv[3], argv[4]
n = int(argv[5]) if len(argv) > 5 else 12
x = pairs(split, [country]).filter((pl.col("h") == H) & (pl.col("nd") == ND) & (pl.col("n_dup") >= 2))
if split == "train":
    x = x.join(links("train", "dataset/train/train_ground_truth.tsv"), on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
else:
    x = x.with_columns(pl.lit(-1).alias("y"))
s1 = R(f"dataset/{split}/{split}_source1.tsv", ["business_name", "business_address"])
tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id", "business_name", "business_address"]) for i in (2, 3)])
x = x.sample(min(n, x.height), seed=7)
for si, ti, y in x.select("si", "ti", "y").iter_rows():
    a = s1.row(si)
    b = tg.row(ti)
    print(f"y={y} | {a[0][:34]:34} {a[1][:40]:40} || {b[0]} {b[1][:34]:34} {b[1][:40]}")
