"""Show France pairs removed by the new hybrid for chosen generator classes, with both addresses and scores."""
import sys

import polars as pl

R = lambda p: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)
s1 = R("dataset/test/test_source1.tsv").with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti")


def pairs(path):
    return (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids").join(s1.select("entity_id", "si", "country"), left_on="source1_entity_id", right_on="entity_id")
            .join(tg.select("entity_id", "ti"), left_on="matched_entity_ids", right_on="entity_id").select("si", "ti", "country"))


base, new = pairs(sys.argv[1]), pairs(sys.argv[2])
removed = base.join(new, on=["si", "ti"], how="anti").filter(pl.col("country") == (sys.argv[4] if len(sys.argv) > 4 else "France"))
gen = pl.read_parquet("cache/gen/test.parquet")
old_cal = pl.read_parquet(sys.argv[1].replace("matching_results.tsv", "test_calibrated.parquet")).rename({"p": "p_old", "raw": "raw_old"})
new_cal = pl.read_parquet(sys.argv[2].replace("matching_results.tsv", "test_calibrated.parquet")).rename({"p": "p_new", "raw": "raw_new"})
feat = pl.concat([pl.read_parquet(p, columns=["si", "ti", "house_match", "house_conflict", "address_idf_cos"])
                  .join(removed.select("si", "ti"), on=["si", "ti"], how="semi") for p in sorted(__import__("glob").glob("cache/v3/test/part*.parquet"))])
x = removed.join(gen, on=["si", "ti"], how="left").join(old_cal, on=["si", "ti"], how="left").join(new_cal, on=["si", "ti"], how="left").join(feat, on=["si", "ti"], how="left")
for col in sys.argv[3].split(","):
    c = x.filter(pl.col(col) == 1)
    print(f"\n== removed {col}: {c.height:,}; house_match {c['house_match'].mean():.2f}, house_conflict {c['house_conflict'].mean():.2f}, addr_cos {c['address_idf_cos'].mean():.2f}")
    print("   raw_new quantiles", [round(v, 3) for v in c["raw_new"].fill_null(0).quantile([0.1, 0.5, 0.9]) if False] or c["raw_new"].fill_null(-1).describe().filter(pl.col("statistic").is_in(["mean", "25%", "50%", "75%"])).rows())
    s = c.sample(min(14, c.height), seed=2).join(s1.select("si", pl.col("business_name").alias("sn"), pl.col("business_address").alias("sa")), on="si").join(
        tg.select("ti", pl.col("business_name").alias("tn"), pl.col("business_address").alias("ta")), on="ti")
    for r in s.iter_rows(named=True):
        print(f"  old={r['p_old'] or 0:.2f}/{r['raw_old'] or 0:.2f} new={r['p_new'] or 0:.2f}/{r['raw_new'] or 0:.2f} hm={r['house_match']} hc={r['house_conflict']}"
              f" | {r['sn'][:30]:30} | {r['sa'][:38]:38} || {r['tn'][:30]:30} | {r['ta'][:40]}")
