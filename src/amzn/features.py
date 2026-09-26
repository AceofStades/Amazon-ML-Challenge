"""Numeric pair features computed inside DuckDB for parallel execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from .blocking import connect


FEATURE_NAMES = (
    "block_weight", "matching_keys", "candidate_rank", "name_jw",
    "address_jw", "name_core_jw", "address_core_jw", "name_exact",
    "address_exact", "name_anchor_overlap", "address_anchor_overlap",
    "house_match", "house_conflict", "name_length_ratio", "address_length_ratio",
    "name_missing", "address_missing", "target_is_source3",
    "name_bigram_jaccard", "address_bigram_jaccard", "name_token_jaccard",
    "address_token_jaccard", "name_edit_similarity", "name_compact_jw",
    "name_anchor_jw", "address_anchor_jw", "address_number_jaccard",
)

SECOND_STAGE_FEATURE_NAMES = FEATURE_NAMES + (
    "seed_score", "seed_name_jw", "seed_address_jw",
)


def create_features(
    connection: duckdb.DuckDBPyConnection, split: str, candidates: str, output: str,
    include_seed_features: bool = False,
) -> None:
    assert split in {"train", "test"}
    assert candidates.replace("_", "").isalnum()
    assert output.replace("_", "").isalnum()
    seed_fields = (
        ", coalesce(c.seed_score, 0)::FLOAT AS seed_score"
        ", coalesce(c.seed_name_jw, 0)::FLOAT AS seed_name_jw"
        ", coalesce(c.seed_address_jw, 0)::FLOAT AS seed_address_jw"
        if include_seed_features else ""
    )
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
               (s.name IS NULL OR s.name = '' OR t.name IS NULL OR t.name = '')::INT::FLOAT AS name_missing,
               (s.address IS NULL OR s.address = '' OR t.address IS NULL OR t.address = '')::INT::FLOAT AS address_missing,
               starts_with(t.entity_id, 'S3-')::INT::FLOAT AS target_is_source3,
               coalesce(jaccard(s.name_core, t.name_core), 0)::FLOAT AS name_bigram_jaccard,
               CASE WHEN s.address <> '' AND t.address <> ''
                    THEN coalesce(jaccard(s.address_core, t.address_core), 0)
                    ELSE 0 END::FLOAT AS address_bigram_jaccard,
               (list_unique(list_intersect(string_split(s.name_core, ' '), string_split(t.name_core, ' ')))::FLOAT /
                greatest(list_unique(string_split(s.name_core, ' ')) +
                         list_unique(string_split(t.name_core, ' ')) -
                         list_unique(list_intersect(string_split(s.name_core, ' '), string_split(t.name_core, ' '))), 1))
                    AS name_token_jaccard,
               CASE WHEN s.address <> '' AND t.address <> '' THEN
                   list_unique(list_intersect(string_split(s.address_core, ' '), string_split(t.address_core, ' ')))::FLOAT /
                   greatest(list_unique(string_split(s.address_core, ' ')) +
                            list_unique(string_split(t.address_core, ' ')) -
                            list_unique(list_intersect(string_split(s.address_core, ' '), string_split(t.address_core, ' '))), 1)
                   ELSE 0 END AS address_token_jaccard,
               (1 - levenshtein(s.name_core, t.name_core)::FLOAT /
                greatest(length(s.name_core), length(t.name_core), 1)) AS name_edit_similarity,
               coalesce(jaro_winkler_similarity(replace(s.name_core, ' ', ''),
                                                replace(t.name_core, ' ', '')), 0)::FLOAT AS name_compact_jw,
               greatest(
                   coalesce(jaro_winkler_similarity(nullif(s.name_anchor1, ''), nullif(t.name_anchor1, '')), 0),
                   coalesce(jaro_winkler_similarity(nullif(s.name_anchor1, ''), nullif(t.name_anchor2, '')), 0),
                   coalesce(jaro_winkler_similarity(nullif(s.name_anchor2, ''), nullif(t.name_anchor1, '')), 0),
                   coalesce(jaro_winkler_similarity(nullif(s.name_anchor2, ''), nullif(t.name_anchor2, '')), 0)
               )::FLOAT AS name_anchor_jw,
               greatest(
                   coalesce(jaro_winkler_similarity(nullif(s.address_anchor1, ''), nullif(t.address_anchor1, '')), 0),
                   coalesce(jaro_winkler_similarity(nullif(s.address_anchor1, ''), nullif(t.address_anchor2, '')), 0),
                   coalesce(jaro_winkler_similarity(nullif(s.address_anchor2, ''), nullif(t.address_anchor1, '')), 0),
                   coalesce(jaro_winkler_similarity(nullif(s.address_anchor2, ''), nullif(t.address_anchor2, '')), 0)
               )::FLOAT AS address_anchor_jw,
               CASE WHEN s.address <> '' AND t.address <> '' THEN
                   list_unique(list_intersect(regexp_extract_all(s.address, '[0-9]+'),
                                              regexp_extract_all(t.address, '[0-9]+')))::FLOAT /
                   greatest(list_unique(regexp_extract_all(s.address, '[0-9]+')) +
                            list_unique(regexp_extract_all(t.address, '[0-9]+')) -
                            list_unique(list_intersect(regexp_extract_all(s.address, '[0-9]+'),
                                                       regexp_extract_all(t.address, '[0-9]+'))), 1)
                   ELSE 0 END AS address_number_jaccard,
               CASE WHEN s.house <> '' AND t.house <> ''
                    THEN levenshtein(s.house, t.house) ELSE -1 END::FLOAT AS house_edit,
               CASE WHEN s.house <> '' AND t.house <> ''
                    THEN (starts_with(s.house, t.house) OR starts_with(t.house, s.house)
                          OR ends_with(s.house, t.house) OR ends_with(t.house, s.house))::INT
                    ELSE -1 END::FLOAT AS house_affix,
               CASE WHEN s.house <> '' AND t.house <> ''
                    THEN abs(length(s.house) - length(t.house)) ELSE -1 END::FLOAT AS house_length_diff
               {seed_fields}
        FROM {candidates} c
        JOIN {split}_source1 s ON c.source1_entity_id = s.entity_id
        JOIN {split}_targets t ON c.candidate_entity_id = t.entity_id
    """)
    count = connection.execute(f"SELECT count(*) FROM {output}").fetchone()[0]
    print(f"{output}: {count:,} scored pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-seed-features", action="store_true")
    args = parser.parse_args()
    connection = connect(args.database, args.cache_dir)
    create_features(connection, args.split, args.candidates, args.output,
                    args.include_seed_features)


if __name__ == "__main__":
    main()
