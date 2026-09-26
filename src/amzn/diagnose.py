"""Per-country profile of a matching_results.tsv file (no labels needed).

Compares predicted matches per Source 1 entity, empty-list share, and the share
of target records claimed, across countries. On training data the matched
share of targets is 0.74 and 5.6% of Source 1 entities are singletons.
"""

from __future__ import annotations

import argparse

import polars as pl


def read(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matching", nargs="+")
    parser.add_argument("--split-dir", default="dataset/test")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    source1 = read(f"{args.split_dir}/{args.split}_source1.tsv").select("entity_id", "country")
    targets = pl.concat([read(f"{args.split_dir}/{args.split}_source{i}.tsv").select("country")
                         for i in (2, 3)]).group_by("country").agg(pl.len().alias("targets"))
    for path in args.matching:
        frame = read(path).join(source1, left_on="source1_entity_id", right_on="entity_id")
        frame = frame.with_columns(
            pl.when(pl.col("matched_entity_ids") == "").then(0)
            .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("k"))
        summary = (
            frame.group_by("country").agg(
                pl.len().alias("entities"), (pl.col("k") == 0).mean().alias("empty"),
                pl.col("k").mean().alias("mean_k"), pl.col("k").sum().alias("matches"),
                *[(pl.col("k") == i).mean().alias(f"k{i}") for i in range(1, 8)],
            )
            .join(targets, on="country")
            .with_columns((pl.col("matches") / pl.col("targets")).alias("claimed_share"))
            .sort("country")
        )
        print(path)
        with pl.Config(tbl_cols=20, tbl_width_chars=200, float_precision=3):
            print(summary)


if __name__ == "__main__":
    main()
