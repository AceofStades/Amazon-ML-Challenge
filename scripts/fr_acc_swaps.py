import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")
fr = s1.filter(pl.col("country") == "France").select("si")
links = (R("output/final_x/matching_results.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti")).join(fr, on="si", how="semi")
gen = pl.read_parquet("cache/gen/test.parquet")
x = links.join(gen, on=["si", "ti"]).filter(pl.col("g_swap_common") == 1)
cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos"]
f = pl.concat([pl.read_parquet(p, columns=cols).join(x.select("si", "ti"), on=["si", "ti"], how="semi") for p in sorted(glob.glob("cache/v3/test/part*.parquet"))])
x = x.join(f, on=["si", "ti"], how="left")
sc = pl.read_parquet("output/blend_pre1/test_scored.parquet", columns=["si", "ti", "p1", "p2"])
x = x.join(sc, on=["si", "ti"], how="left")
print("accepted France common-word swaps:", x.height, "same house:", (x["house_match"] == 1).sum(), "typo flag:", (x["g_typo1"] == 1).sum(),
      "p2 > 0.98:", (x["p2"] > 0.98).sum(), "p2 < 0.1:", (x["p2"] < 0.1).sum())
s = x.sample(20, seed=1).join(s1.select("si", pl.col("business_name").alias("sn")), on="si").join(tg.select("ti", pl.col("business_name").alias("tn")), on="ti")
for r in s.iter_rows(named=True):
    print(f"  h={r['house_match']} p1={r['p1']:.2f} p2={r['p2']:.2f} typo={r['g_typo1']} | {r['sn'][:34]:34} || {r['tn'][:34]}")
