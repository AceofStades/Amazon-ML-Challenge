"""Numeric pair features computed inside DuckDB for parallel execution."""

from __future__ import annotations

import duckdb


FEATURE_NAMES = (
    "block_weight", "matching_keys", "candidate_rank", "name_jw",
    "address_jw", "name_core_jw", "address_core_jw", "name_exact",
    "address_exact", "name_anchor_overlap", "address_anchor_overlap",
    "house_match", "house_conflict", "name_length_ratio", "address_length_ratio",
    "name_missing", "address_missing", "target_is_source3",
)


def create_features(
    connection: duckdb.DuckDBPyConnection, split: str, candidates: str, output: str
) -> None:
    assert split in {"train", "test"}
    assert candidates.replace("_", "").isalnum()
    assert output.replace("_", "").isalnum()
    connection.execute(f"""
        CREATE OR REPLACE TABLE {output} AS
        SELECT c.source1_entity_id, c.candidate_entity_id,
               c.block_weight::FLOAT AS block_weight,
               c.matching_keys::FLOAT AS matching_keys,
               c.candidate_rank::FLOAT AS candidate_rank,
               c.name_jw::FLOAT AS name_jw,
               c.address_jw::FLOAT AS address_jw,
               jaro_winkler_similarity(s.name_core, t.name_core)::FLOAT AS name_core_jw,
               CASE WHEN s.address <> '' AND t.address <> ''
                    THEN jaro_winkler_similarity(s.address_core, t.address_core)
                    ELSE 0 END::FLOAT AS address_core_jw,
               (s.name_core <> '' AND s.name_core = t.name_core)::INT::FLOAT AS name_exact,
               (s.address_core <> '' AND s.address_core = t.address_core)::INT::FLOAT AS address_exact,
               (s.name_anchor1 <> '' AND
                (s.name_anchor1 = t.name_anchor1 OR s.name_anchor1 = t.name_anchor2)
                OR s.name_anchor2 <> '' AND
                (s.name_anchor2 = t.name_anchor1 OR s.name_anchor2 = t.name_anchor2))::INT::FLOAT
                AS name_anchor_overlap,
               (s.address_anchor1 <> '' AND
                (s.address_anchor1 = t.address_anchor1 OR s.address_anchor1 = t.address_anchor2)
                OR s.address_anchor2 <> '' AND
                (s.address_anchor2 = t.address_anchor1 OR s.address_anchor2 = t.address_anchor2))::INT::FLOAT
                AS address_anchor_overlap,
               (s.house <> '' AND s.house = t.house)::INT::FLOAT AS house_match,
               (s.house <> '' AND t.house <> '' AND s.house <> t.house)::INT::FLOAT AS house_conflict,
               (least(length(s.name), length(t.name))::FLOAT /
                greatest(length(s.name), length(t.name), 1)) AS name_length_ratio,
               (least(length(s.address), length(t.address))::FLOAT /
                greatest(length(s.address), length(t.address), 1)) AS address_length_ratio,
               (s.name = '' OR t.name = '')::INT::FLOAT AS name_missing,
               (s.address = '' OR t.address = '')::INT::FLOAT AS address_missing,
               starts_with(t.entity_id, 'S3-')::INT::FLOAT AS target_is_source3
        FROM {candidates} c
        JOIN {split}_source1 s ON c.source1_entity_id = s.entity_id
        JOIN {split}_targets t ON c.candidate_entity_id = t.entity_id
    """)
    count = connection.execute(f"SELECT count(*) FROM {output}").fetchone()[0]
    print(f"{output}: {count:,} scored pairs", flush=True)
