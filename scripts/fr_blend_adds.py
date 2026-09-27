"""France pairs accepted by the blend-scored France hybrid but not by the current France decisions: house relation,
stage-1 p1, blend p2, and names/addresses, to judge whether a union would add true links."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")


def pairs(path):
    return (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
            .filter(pl.col("country") == "France")
            .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))


base, new = pairs("output/final_pre1/matching_results.tsv"), pairs("output/hybrid_auto2_frblend/matching_results.tsv")
added = new.join(base, on=["si", "ti"], how="anti")
p1 = pl.concat([pl.read_parquet(f).join(added, on=["si", "ti"], how="semi") for f in sorted(glob.glob("output/v7/test_p1/*.parquet"))])
p2 = pl.read_parquet("output/hybrid_auto2_frblend/test_calibrated.parquet").join(added, on=["si", "ti"], how="semi")
gen = pl.read_parquet("cache/gen/test.parquet").join(added, on=["si", "ti"], how="semi")
feat = pl.concat([pl.read_parquet(f, columns=["si", "ti", "house_match", "address_idf_cos", "name_idf_cos"]).join(added, on=["si", "ti"], how="semi")
                  for f in sorted(glob.glob("cache/v3/test/part*.parquet"))])
x = added.join(p1, on=["si", "ti"], how="left").join(p2.rename({"p": "p_cal", "raw": "p2"}), on=["si", "ti"], how="left") \
         .join(feat, on=["si", "ti"], how="left").join(gen.select("si", "ti", "g_exact"), on=["si", "ti"], how="left")
# Is the target already assigned to another France entity in the current decisions?
x = x.join(base.select("ti", pl.col("si").alias("other_si")), on="ti", how="left")
print("added:", x.height, "target taken by another entity in base:", x["other_si"].is_not_null().sum())
print(x.group_by("house_match").agg(pl.len(), pl.col("p1").median().alias("p1_med"), pl.col("p2").median().alias("p2_med"),
                                    pl.col("address_idf_cos").median().alias("addr_cos"), pl.col("g_exact").mean().alias("exact")).sort("house_match"))
ex = x.filter(pl.col("g_exact") == 1).sample(15, seed=3).join(s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si") \
      .join(tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
for r in ex.iter_rows(named=True):
    print(f"h={r['house_match']} p1={r['p1']:.2f} p2={r['p2']:.2f} other={r['other_si'] is not None} | {r['sn'][:28]:28} {r['sa'][:38]:38} || {r['tn'][:28]:28} {r['ta'][:38]}")
