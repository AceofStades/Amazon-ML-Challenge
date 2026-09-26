"""Audit addressed France candidates rejected by the current rules.

This diagnostic never changes matching decisions. Run under logs/blend.lock.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import polars as pl


def read(path: str, columns: list[str]) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       missing_utf8_is_empty_string=True, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="output/sub_auto2_frsafe/matching_results.tsv")
    parser.add_argument("--output", default="output/france_audit_rejected.parquet")
    args = parser.parse_args()
    s_ids = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").with_row_index("si")
    fr = s_ids.filter(pl.col("country") == "France").select("si", "entity_id")
    t_ids = pl.concat([pl.read_parquet(f"cache/embeddings/test_source{i}_ids.parquet")
                      for i in (2, 3)]).with_row_index("ti")
    acc = (read(args.baseline, ["source1_entity_id", "matched_entity_ids"])
           .join(fr, left_on="source1_entity_id", right_on="entity_id")
           .filter(pl.col("matched_entity_ids") != "")
           .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
           .join(t_ids.select("ti", "entity_id"), left_on="matched_entity_ids", right_on="entity_id")
           .select("si", "ti").with_columns(pl.lit(True).alias("accepted")))
    columns = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "address_exact",
               "name_idf_cos", "name_compact_jw", "name_exact", "s_name_dup", "t_name_dup"]
    chunks = []
    for path in sorted(glob.glob("cache/v3/test/part*.parquet")):
        feats = (pl.scan_parquet(path).select(columns)
                 .join(fr.select("si").lazy(), on="si", how="semi")
                 .filter((pl.col("address_idf_cos") >= 0.6)
                         & ((pl.col("house_match") == 1) | (pl.col("address_idf_cos") >= 0.9))))
        frame = feats.join(pl.scan_parquet(f"output/v7/test_p1/{Path(path).name}"),
                           on=["si", "ti"]).collect()
        chunks.append(frame)
        print(f"{Path(path).name}: {frame.height:,}", flush=True)
    frame = pl.concat(chunks).join(acc, on=["si", "ti"], how="left").with_columns(
        pl.col("accepted").fill_null(False)).join(
            acc.select("ti").unique().with_columns(pl.lit(True).alias("target_taken")), on="ti", how="left")
    frame = frame.with_columns(pl.col("target_taken").fill_null(False))
    gen = pl.read_parquet("cache/gen/test.parquet").join(fr.select("si"), on="si", how="semi")
    frame = frame.join(gen, on=["si", "ti"], how="left")
    classes = [("acronym", "g_acronym"), ("domain", "g_domain"), ("alias", "g_alias"),
               ("common_swap", "g_swap_common"), ("filler", "g_swap_filler"), ("rare_swap", "g_swap_rare"),
               ("typo", "g_typo1"), ("brand", "g_brand1"), ("exact", "g_exact"),
               ("add", "g_add"), ("drop", "g_drop"), ("multi", "g_multi")]
    expr = pl.lit("unscored")
    for name, column in reversed(classes):
        expr = pl.when(pl.col(column) == 1).then(pl.lit(name)).otherwise(expr)
    frame = frame.with_columns(expr.alias("class"))
    print("SUMMARY", flush=True)
    summary = (frame.group_by("class", "accepted", "target_taken").agg(
        pl.len().alias("pairs"), pl.col("ti").n_unique().alias("targets"),
        (pl.col("p1") < 0.01).sum().alias("p_lt_01"),
        pl.col("p1").median().round(4).alias("p_median"),
        (pl.col("address_idf_cos") >= 0.9).sum().alias("address_ge_9"),
        (pl.col("address_exact") == 1).sum().alias("address_exact"))
        .sort(["accepted", "target_taken", "pairs"], descending=[False, False, True]))
    for row in summary.iter_rows(named=True):
        print(json.dumps(row), flush=True)
    rejected = frame.filter(~pl.col("target_taken"))
    s = read("cache/test_source1.tsv", ["name_core", "name", "address", "house"]).with_row_index("si")
    t = pl.concat([read(f"cache/test_source{i}.tsv", ["name_core", "name", "address", "house"])
                   for i in (2, 3)]).with_row_index("ti")
    rejected = (rejected.join(s.rename({k: f"s_{k}" for k in s.columns if k != "si"}), on="si")
                .join(t.rename({k: f"t_{k}" for k in t.columns if k != "ti"}), on="ti"))
    rejected.write_parquet(args.output)
    print(f"Saved {rejected.height:,} unassigned candidates to {args.output}", flush=True)
    for cls in ("unscored", "exact", "drop", "add", "filler", "rare_swap", "typo", "multi"):
        group = rejected.filter(pl.col("class") == cls)
        print(f"SAMPLES {cls} n={group.height}", flush=True)
        for row in group.sort("p1", descending=True).head(8).iter_rows(named=True):
            print(json.dumps({key: row[key] for key in ["si", "ti", "p1", "address_idf_cos",
                  "house_match", "house_conflict", "s_name", "t_name", "s_address", "t_address"]}), flush=True)


if __name__ == "__main__":
    main()
