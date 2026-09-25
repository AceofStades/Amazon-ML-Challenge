"""Inspect positive training pairs missed by candidate generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from .blocking import sql_string


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--sample-per-thousand", type=int, default=60)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()
    connection = duckdb.connect(str(args.database), read_only=True)
    query = f"""
        WITH truth AS (
            SELECT source1_entity_id,
                   unnest(string_split(matched_entity_ids, ',')) AS candidate_entity_id
            FROM read_csv('{sql_string(args.truth)}', delim='\\t', header=true, all_varchar=true)
            WHERE hash(source1_entity_id) % 1000 < {int(args.sample_per_thousand)}
              AND matched_entity_ids <> ''
        )
        SELECT t.source1_entity_id, t.candidate_entity_id,
               s.name AS source_name, x.name AS target_name,
               s.address AS source_address, x.address AS target_address,
               s.country AS source_country, x.country AS target_country,
               jaro_winkler_similarity(s.name, x.name) AS name_jw,
               jaro_winkler_similarity(s.address, x.address) AS address_jw,
               c.candidate_rank
        FROM truth t
        JOIN train_source1 s ON s.entity_id = t.source1_entity_id
        JOIN train_targets x ON x.entity_id = t.candidate_entity_id
        LEFT JOIN train_sample_candidates c USING (source1_entity_id, candidate_entity_id)
    """
    for row in connection.execute(f"""
        SELECT candidate_rank IS NOT NULL AS found, count(*) AS pairs,
               avg((source_country = target_country)::INT) AS country_agreement,
               avg(name_jw) AS name_jw, avg(address_jw) AS address_jw,
               avg((source_address = '' OR target_address = '')::INT) AS address_missing
        FROM ({query}) GROUP BY 1 ORDER BY 1
    """).fetchall():
        print(row)
    for row in connection.execute(f"""
        SELECT * FROM ({query}) WHERE candidate_rank IS NULL
        ORDER BY hash(source1_entity_id || candidate_entity_id)
        LIMIT {int(args.examples)}
    """).fetchall():
        print(row)


if __name__ == "__main__":
    main()
