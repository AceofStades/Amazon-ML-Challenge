"""France accepted links whose addresses are on 'another street' (address IDF cosine < 0.6) with the same house number:
samples and the street-token overlap, to tell reformatted addresses from genuinely different streets."""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
sub = sys.argv[1] if len(sys.argv) > 1 else "output/final_ctxA/matching_results.tsv"
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
col = "matched_entity_ids"
l = (R(sub).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
     .join(s1.filter(pl.col("country") == "France").select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
     .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti"))
f = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]).join(l, on=["si", "ti"], how="semi")
               for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
x = f.filter((pl.col("address_idf_cos") < 0.6) & (pl.col("house_match") == 1))
print(f"France accepted links, other street + same house: {x.height:,} of {l.height:,}")
print(x.with_columns(pl.col("address_idf_cos").cut([0.2, 0.3, 0.4, 0.5]).alias("b")).group_by("b").len().sort("b"))
sa = R("cache/test_source1.tsv", ["address"])["address"]
ta = pl.concat([R(f"cache/test_source{i}.tsv", ["address"]) for i in (2, 3)])["address"]
x = x.with_columns(s1["business_name"].gather(x["si"].to_numpy()).alias("sn"), tg["business_name"].gather(x["ti"].to_numpy()).alias("tn"),
                   s1["business_address"].gather(x["si"].to_numpy()).alias("sa"), tg["business_address"].gather(x["ti"].to_numpy()).alias("ta"),
                   sa.gather(x["si"].to_numpy()).alias("sa_n"), ta.gather(x["ti"].to_numpy()).alias("ta_n"))
for r in x.sample(25, seed=11).iter_rows(named=True):
    print(f"{r['address_idf_cos']:.2f} | {r['sa'][:58]:58} || {r['ta'][:62]}")
x.select("si", "ti", "address_idf_cos", "sa_n", "ta_n").write_parquet("cache/fr_otherstreet_links.parquet")
