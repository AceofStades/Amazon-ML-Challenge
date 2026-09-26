"""Combine submissions: US/India rows from one file, France rows from another, plus optional France additions
from a third file restricted to name-change classes that are true noise in training (safe classes).

  python scripts/merge_france.py --usin A.tsv --france B.tsv [--france-extra C.tsv] --output OUT.tsv
"""

import argparse

import polars as pl

SAFE = ["g_drop", "g_swap_filler", "g_exact", "g_domain", "g_typo1", "g_acronym", "g_add"]


def read(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True)


def pairs(frame: pl.DataFrame) -> pl.DataFrame:
    return (frame.filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usin", required=True)
    parser.add_argument("--france", required=True)
    parser.add_argument("--france-extra")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    s1 = read("dataset/test/test_source1.tsv").with_row_index("si").select("entity_id", "si", "country")
    targets = pl.concat([read(f"dataset/test/test_source{i}.tsv") for i in (2, 3)]).with_row_index("ti").select("entity_id", "ti")
    fr_ids = s1.filter(pl.col("country") == "France").select(pl.col("entity_id").alias("source1_entity_id"))
    usin = pairs(read(args.usin)).join(fr_ids, on="source1_entity_id", how="anti")
    france = pairs(read(args.france)).join(fr_ids, on="source1_entity_id", how="semi")
    merged = pl.concat([usin, france])
    if args.france_extra:
        extra = (pairs(read(args.france_extra)).join(fr_ids, on="source1_entity_id", how="semi")
                 .join(france, on=["source1_entity_id", "matched_entity_ids"], how="anti"))
        gen = pl.read_parquet("cache/gen/test.parquet", columns=["si", "ti", *SAFE])
        extra = (extra.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "si"), on="source1_entity_id")
                 .join(targets.select(pl.col("entity_id").alias("matched_entity_ids"), "ti"), on="matched_entity_ids")
                 .join(gen, on=["si", "ti"], how="left"))
        safe = extra.filter(pl.any_horizontal([pl.col(c).fill_null(0) == 1 for c in SAFE]))
        # One Source 1 entity per target: skip targets the France set already assigns.
        safe = safe.join(merged.select("matched_entity_ids"), on="matched_entity_ids", how="anti")
        print(f"France extra pairs {extra.height:,}; safe-class additions {safe.height:,}", flush=True)
        merged = pl.concat([merged, safe.select("source1_entity_id", "matched_entity_ids")])
    lists = merged.group_by("source1_entity_id").agg(pl.col("matched_entity_ids").unique().sort().str.join(","))
    out = s1.select(pl.col("entity_id").alias("source1_entity_id")).join(lists, on="source1_entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").fill_null(""))
    out.write_csv(args.output, separator="\t", quote_style="never")
    print(f"wrote {args.output}: {out.height:,} rows, {merged.height:,} matches", flush=True)


if __name__ == "__main__":
    main()
