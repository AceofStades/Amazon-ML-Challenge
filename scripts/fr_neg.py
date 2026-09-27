"""France pairs accepted by file 11 that the blend strongly rejects (p2 < 0.05), by generator class and house relation,
with samples: candidates for France false positives (except French-suffix noise, which the blend does not know)."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
links = (R("output/final_pre1_frcu/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti")).join(fr, on="si", how="semi")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"])
x = links.join(sc, on=["si", "ti"], how="left")
print("France accepted:", x.height, "without blend score:", x["p2"].is_null().sum(), "p2<0.05:", (x["p2"] < 0.05).sum(), "p2<0.5:", (x["p2"] < 0.5).sum())
gen = pl.read_parquet("cache/gen/test.parquet")
classes = [("acronym", "g_acronym"), ("domain", "g_domain"), ("alias", "g_alias"), ("brand1", "g_brand1"), ("swap_common", "g_swap_common"),
           ("swap_filler", "g_swap_filler"), ("swap_rare", "g_swap_rare"), ("typo1", "g_typo1"), ("exact", "g_exact"), ("add", "g_add"), ("drop", "g_drop"), ("multi", "g_multi")]
lab = pl.lit("other")
for name, col in reversed(classes):
    lab = pl.when(pl.col(col) == 1).then(pl.lit(name)).otherwise(lab)
lo = x.filter(pl.col("p2") < 0.05).join(gen, on=["si", "ti"], how="left").with_columns(lab.alias("cls"))
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos"]
f = pl.concat([pl.read_parquet(p, columns=cols).join(lo.select("si", "ti"), on=["si", "ti"], how="semi") for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
lo = lo.join(f, on=["si", "ti"], how="left").join(tg.select("ti", (pl.col("business_address") == "").alias("noaddr")), on="ti")
lo = lo.with_columns(pl.when(pl.col("noaddr")).then(pl.lit("noaddr")).when(pl.col("address_idf_cos") < 0.6).then(pl.lit("otherst"))
                     .when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown")).alias("where"))
pl.Config.set_tbl_rows(60)
print(lo.group_by("cls", "where").len().pivot(on="where", index="cls", values="len").fill_null(0).sort("cls"))
for c in ("exact", "drop", "add", "other", "swap_rare"):
    s = lo.filter(pl.col("cls") == c)
    s = s.sample(min(6, s.height), seed=1).join(s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
        tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
    print("---", c)
    for r in s.iter_rows(named=True):
        print(f"  {r['where']:8} p1={r['p1']:.2f} p2={r['p2']:.3f} | {r['sn'][:28]:28} {r['sa'][:30]:30} || {r['tn'][:28]:28} {r['ta'][:30]}")
