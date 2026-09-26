"""Compare matching_results files: matches per entity by country, and France flips by generator-feature class."""
import sys

import polars as pl

R = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
s1 = R("dataset/test/test_source1.tsv").with_row_index("si").select("entity_id", "si", "country", pl.col("business_name").alias("s_name"))
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti").select(
    "entity_id", "ti", pl.col("business_name").alias("t_name"), pl.col("business_address").alias("t_addr"))


def pairs(path):
    return (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
            .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti", "country"))


base, new = pairs(sys.argv[1]), pairs(sys.argv[2])
E = s1.group_by("country").len().rename({"len": "E"})
for name, frame in (("base", base), ("new", new)):
    print(name, frame.group_by("country").len().join(E, on="country").with_columns((pl.col("len") / pl.col("E")).round(4)).sort("country").rows())
added = new.join(base, on=["si", "ti"], how="anti")
removed = base.join(new, on=["si", "ti"], how="anti")
print("added by country:", added.group_by("country").len().sort("country").rows())
print("removed by country:", removed.group_by("country").len().sort("country").rows())
gen = pl.read_parquet("cache/gen/test.parquet")
classes = [("swap_common", "g_swap_common"), ("swap_filler", "g_swap_filler"), ("swap_rare", "g_swap_rare"), ("typo1", "g_typo1"),
           ("acronym", "g_acronym"), ("domain", "g_domain"), ("brand1", "g_brand1"), ("exact", "g_exact"), ("add", "g_add"),
           ("drop", "g_drop"), ("multi", "g_multi")]
label = pl.lit("other")
for name, col in reversed(classes):
    label = pl.when(pl.col(col) == 1).then(pl.lit(name)).otherwise(label)
country = sys.argv[3] if len(sys.argv) > 3 else "France"
for tag, frame in (("added", added), ("removed", removed)):
    f = frame.filter(pl.col("country") == country).join(gen, on=["si", "ti"], how="left").with_columns(label.alias("cls"))
    print(f"{country} {tag}: {f.height:,}", f.group_by("cls").len().sort("len", descending=True).rows())
    sample = f.sample(min(12, f.height), seed=4).join(s1.select("si", "s_name"), on="si").join(tg.select("ti", "t_name", "t_addr"), on="ti")
    for r in sample.iter_rows(named=True):
        print(f"   [{r['cls']}] {r['s_name'][:38]:38} || {r['t_name'][:38]:38} | {r['t_addr'][:40]}")
