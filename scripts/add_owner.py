"""Add address-less owner-model assignments (cache/noaddr_owner_test.parquet from scripts/noaddr_owner.py) with model
probability >= --min-pm to a submission, for targets no entity holds.
Usage: python scripts/add_owner.py --base A/matching_results.tsv --output B/matching_results.tsv [--min-pm 0.9]"""
import argparse

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--min-pm", type=float, default=0.9)
args = ap.parse_args()
s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
links = (R(args.base).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
own = pl.read_parquet("cache/noaddr_owner_test.parquet").filter(pl.col("pm") >= args.min_pm).join(links.select("ti").unique(), on="ti", how="anti")
print(f"owner-model additions with pm >= {args.min_pm}: {own.height:,}", own.group_by("country").len().sort("country").rows())
new = pl.concat([links, own.select("si", "ti")]).join(s1.select("si", pl.col("entity_id").alias("source1_entity_id")), on="si").join(
    tg.select("ti", pl.col("entity_id").alias("t")), on="ti")
grouped = new.group_by("source1_entity_id").agg(pl.col("t").sort().str.join(",").alias("matched_entity_ids"))
out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(grouped, on="source1_entity_id", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
out.write_csv(args.output, separator="\t", quote_style="never")
print(f"wrote {args.output}: {out.height:,} rows, {new.height:,} matches (base {links.height:,})")
