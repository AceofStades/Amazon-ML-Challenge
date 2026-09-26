"""Find France name-noise candidates with shortened or altered addresses."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import polars as pl

from france_audit_restore import eligible


def read(path: str, columns: list[str]) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       empty_string_is_null=False, columns=columns)


def main() -> None:
    source_ids = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").with_row_index("si")
    france = source_ids.filter(pl.col("country") == "France").select("si", "entity_id")
    target_ids = pl.concat([pl.read_parquet(f"cache/embeddings/test_source{i}_ids.parquet")
                            for i in (2, 3)]).with_row_index("ti")
    accepted = (read("output/sub_auto2_frsafe/matching_results.tsv", ["source1_entity_id", "matched_entity_ids"])
                .join(france.select("entity_id"), left_on="source1_entity_id", right_on="entity_id")
                .filter(pl.col("matched_entity_ids") != "")
                .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
                .join(target_ids.select("ti", "entity_id"), left_on="matched_entity_ids", right_on="entity_id")
                .select("ti").unique())
    gen = (pl.read_parquet("cache/gen/test.parquet").join(france.select("si"), on="si", how="semi")
           .join(accepted, on="ti", how="anti"))
    gen = gen.filter(eligible(gen))
    columns = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "address_idf_s_cover",
               "address_idf_t_cover", "address_anchor_jw", "address_core_jw", "name_idf_cos", "name_compact_jw"]
    chunks = []
    for path in sorted(glob.glob("cache/v3/test/part*.parquet")):
        frame = (pl.scan_parquet(path).select(columns)
                 .filter((pl.col("house_match") == 1) & (pl.col("address_idf_cos") < 0.8))
                 .join(gen.lazy(), on=["si", "ti"], how="inner")
                 .join(pl.scan_parquet(f"output/v7/test_p1/{Path(path).name}"), on=["si", "ti"]).collect())
        chunks.append(frame)
    frame = pl.concat(chunks)
    source = read("cache/test_source1.tsv", ["name", "address"]).with_row_index("si").join(
        frame.select("si").unique(), on="si", how="semi").rename({"name": "s_name", "address": "s_address"})
    target = pl.concat([read(f"cache/test_source{i}.tsv", ["name", "address"]) for i in (2, 3)]).with_row_index("ti")
    target = target.join(frame.select("ti").unique(), on="ti", how="semi").rename(
        {"name": "t_name", "address": "t_address"})
    frame = frame.join(source, on="si").join(target, on="ti")
    frame.write_parquet("output/france_audit_shortaddr.parquet")
    print(f"Unassigned safe-name France candidates, equal first house, address cosine<.8: {frame.height:,}")
    for coverage in (0.8, 0.9, 0.95):
        for prob in (0.01, 0.5, 0.9, 0.99):
            group = frame.filter((pl.col("address_idf_t_cover") >= coverage) & (pl.col("p1") >= prob))
            print(json.dumps({"target_coverage_min": coverage, "p1_min": prob, "pairs": group.height,
                              "targets": group["ti"].n_unique(),
                              "exact_name": int(group["g_exact"].sum()),
                              "drop_name": int(group["g_drop"].sum()),
                              "filler_name": int(group["g_swap_filler"].sum())}), flush=True)
    for row in frame.sort("p1", descending=True).head(20).iter_rows(named=True):
        print(json.dumps({key: row[key] for key in ["si", "ti", "p1", "g_exact", "g_drop", "g_typo1",
            "address_idf_cos", "address_idf_s_cover", "address_idf_t_cover",
            "s_name", "t_name", "s_address", "t_address"]}), flush=True)


if __name__ == "__main__":
    main()
