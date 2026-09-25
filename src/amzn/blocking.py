"""Disk-backed, bounded blocking with several independent match keys."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def connect(path: Path, cache_dir: Path, threads: int = 16) -> duckdb.DuckDBPyConnection:
    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(f"SET threads = {int(threads)}")
    connection.execute("SET memory_limit = '20GB'")
    connection.execute(f"SET temp_directory = '{sql_string(cache_dir / 'tmp')}'")
    return connection


def sql_string(value: Path | str) -> str:
    return str(value).replace("'", "''")


def import_split(connection: duckdb.DuckDBPyConnection, cache_dir: Path, split: str) -> None:
    """Import normalized TSV once; later runs reuse DuckDB tables."""
    assert split in {"train", "test"}
    for source in (1, 2, 3):
        path = cache_dir / f"{split}_source{source}.tsv"
        if not path.exists():
            raise FileNotFoundError(path)
        connection.execute(
            f"CREATE OR REPLACE TABLE {split}_source{source} AS "
            f"SELECT * FROM read_csv('{sql_string(path)}', delim='\\t', header=true, all_varchar=true)"
        )
        print(f"Imported {split}_source{source}", flush=True)
    connection.execute(
        f"CREATE OR REPLACE TABLE {split}_targets AS "
        f"SELECT * FROM {split}_source2 UNION ALL SELECT * FROM {split}_source3"
    )


def key_query(records: str) -> str:
    """Produce keys. Empty fields never create a key."""
    return f"""
        SELECT r.entity_id, k.key, k.weight, k.max_frequency
        FROM {records} r
        CROSS JOIN LATERAL (VALUES
            (CASE WHEN r.name_core <> '' THEN 'N|' || r.country || '|' || r.name_core END, 100, 500),
            (CASE WHEN r.name_anchor1 <> '' AND r.name_anchor2 <> ''
                  THEN 'NN|' || r.country || '|' || least(r.name_anchor1, r.name_anchor2) || '|' || greatest(r.name_anchor1, r.name_anchor2) END, 90, 1000),
            (CASE WHEN length(r.name_anchor1) >= 4 AND length(r.name_anchor2) >= 4
                  THEN 'NNP|' || r.country || '|' || least(left(r.name_anchor1, 4), left(r.name_anchor2, 4)) || '|' || greatest(left(r.name_anchor1, 4), left(r.name_anchor2, 4)) END, 70, 300),
            (CASE WHEN r.address_core <> '' THEN 'A|' || r.country || '|' || r.address_core END, 95, 300),
            (CASE WHEN r.name_anchor1 <> '' AND r.address_anchor1 <> '' THEN 'NA|' || r.country || '|' || r.name_anchor1 || '|' || r.address_anchor1 END, 75, 300),
            (CASE WHEN r.name_anchor1 <> '' AND r.address_anchor2 <> '' THEN 'NA|' || r.country || '|' || r.name_anchor1 || '|' || r.address_anchor2 END, 72, 300),
            (CASE WHEN r.name_anchor2 <> '' AND r.address_anchor1 <> '' THEN 'NA|' || r.country || '|' || r.name_anchor2 || '|' || r.address_anchor1 END, 70, 300),
            (CASE WHEN r.name_anchor2 <> '' AND r.address_anchor2 <> '' THEN 'NA|' || r.country || '|' || r.name_anchor2 || '|' || r.address_anchor2 END, 68, 300),
            (CASE WHEN length(r.name_anchor1) >= 4 AND r.address_anchor1 <> '' THEN 'NPA|' || r.country || '|' || left(r.name_anchor1, 4) || '|' || r.address_anchor1 END, 60, 300),
            (CASE WHEN length(r.name_anchor1) >= 4 AND r.address_anchor2 <> '' THEN 'NPA|' || r.country || '|' || left(r.name_anchor1, 4) || '|' || r.address_anchor2 END, 58, 300),
            (CASE WHEN length(r.name_anchor2) >= 4 AND r.address_anchor1 <> '' THEN 'NPA|' || r.country || '|' || left(r.name_anchor2, 4) || '|' || r.address_anchor1 END, 56, 300),
            (CASE WHEN length(r.name_anchor2) >= 4 AND r.address_anchor2 <> '' THEN 'NPA|' || r.country || '|' || left(r.name_anchor2, 4) || '|' || r.address_anchor2 END, 54, 300),
            (CASE WHEN r.name_anchor1 <> '' AND length(r.address_anchor1) >= 4 THEN 'APA|' || r.country || '|' || r.name_anchor1 || '|' || left(r.address_anchor1, 4) END, 52, 300),
            (CASE WHEN r.name_anchor1 <> '' AND length(r.address_anchor2) >= 4 THEN 'APA|' || r.country || '|' || r.name_anchor1 || '|' || left(r.address_anchor2, 4) END, 50, 300),
            (CASE WHEN r.name_anchor2 <> '' AND length(r.address_anchor1) >= 4 THEN 'APA|' || r.country || '|' || r.name_anchor2 || '|' || left(r.address_anchor1, 4) END, 48, 300),
            (CASE WHEN r.name_anchor2 <> '' AND length(r.address_anchor2) >= 4 THEN 'APA|' || r.country || '|' || r.name_anchor2 || '|' || left(r.address_anchor2, 4) END, 46, 300),
            (CASE WHEN r.house <> '' AND r.address_anchor1 <> '' THEN 'HA|' || r.country || '|' || r.house || '|' || r.address_anchor1 END, 85, 120),
            (CASE WHEN r.house <> '' AND r.address_anchor2 <> '' THEN 'HA|' || r.country || '|' || r.house || '|' || r.address_anchor2 END, 83, 120),
            (CASE WHEN r.address_anchor1 <> '' AND r.address_anchor2 <> '' THEN 'AA|' || r.country || '|' || r.address_anchor1 || '|' || r.address_anchor2 END, 65, 200),
            (CASE WHEN r.house <> '' AND r.name_anchor1 <> '' THEN 'HN|' || r.country || '|' || r.house || '|' || r.name_anchor1 END, 60, 200),
            (CASE WHEN r.house <> '' AND r.name_anchor2 <> '' THEN 'HN|' || r.country || '|' || r.house || '|' || r.name_anchor2 END, 58, 200),
            (CASE WHEN r.name_anchor1 <> '' THEN 'T|' || r.country || '|' || r.name_anchor1 END, 30, 60),
            (CASE WHEN r.name_anchor2 <> '' THEN 'T|' || r.country || '|' || r.name_anchor2 END, 28, 60),
            (CASE WHEN r.address_anchor1 <> '' THEN 'D|' || r.country || '|' || r.address_anchor1 END, 20, 40),
            (CASE WHEN r.address_anchor2 <> '' THEN 'D|' || r.country || '|' || r.address_anchor2 END, 18, 40)
        ) AS k(key, weight, max_frequency)
        WHERE k.key IS NOT NULL
    """


def index_targets(connection: duckdb.DuckDBPyConnection, split: str) -> None:
    assert split in {"train", "test"}
    connection.execute(
        f"CREATE OR REPLACE TABLE {split}_raw_keys AS {key_query(f'{split}_targets')}"
    )
    print(f"Built raw {split} target keys", flush=True)
    connection.execute(f"""
        CREATE OR REPLACE TABLE {split}_target_keys AS
        SELECT r.entity_id, r.key
        FROM {split}_raw_keys r
        JOIN (
            SELECT key
            FROM {split}_raw_keys
            GROUP BY key, max_frequency
            HAVING count(*) <= max_frequency
        ) good USING (key)
    """)
    connection.execute(f"DROP TABLE {split}_raw_keys")
    print(f"Indexed {split} target keys", flush=True)


def generate_candidates(
    connection: duckdb.DuckDBPyConnection,
    split: str,
    output_table: str,
    source_filter: str = "TRUE",
    max_candidates: int = 30,
) -> None:
    """Keep strongest bounded set immediately before model scoring."""
    assert split in {"train", "test"}
    assert output_table.replace("_", "").isalnum()
    connection.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH source_records AS (
            SELECT * FROM {split}_source1 WHERE {source_filter}
        ),
        source_keys AS (
            {key_query('source_records')}
        ),
        hits AS (
            SELECT s.entity_id AS source1_entity_id, t.entity_id AS candidate_entity_id,
                   max(s.weight) AS block_weight, count(*) AS matching_keys
            FROM source_keys s
            JOIN {split}_target_keys t USING (key)
            GROUP BY 1, 2
        ),
        scored AS (
            SELECT h.*,
                   jaro_winkler_similarity(s.name, t.name) AS name_jw,
                   CASE WHEN s.address <> '' AND t.address <> ''
                        THEN jaro_winkler_similarity(s.address, t.address)
                        ELSE 0 END AS address_jw,
                   coalesce((s.house <> '' AND s.house = t.house)::INT, 0) AS house_equal
            FROM hits h
            JOIN source_records s ON h.source1_entity_id = s.entity_id
            JOIN {split}_targets t ON h.candidate_entity_id = t.entity_id
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY source1_entity_id
                ORDER BY 70 * coalesce(name_jw, 0) + 60 * address_jw + 0.2 * block_weight
                         + 12 * house_equal + 3 * matching_keys DESC,
                         matching_keys DESC, candidate_entity_id
            ) AS candidate_rank
            FROM scored
        )
        SELECT * FROM ranked WHERE candidate_rank <= {int(max_candidates)}
    """)
    count, entities = connection.execute(
        f"SELECT count(*), count(DISTINCT source1_entity_id) FROM {output_table}"
    ).fetchone()
    print(f"{output_table}: {count:,} pairs for {entities:,} Source 1 records", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--import-records", action="store_true")
    parser.add_argument("--index-targets", action="store_true")
    parser.add_argument("--candidates", action="store_true")
    parser.add_argument("--sample-per-thousand", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()

    connection = connect(args.cache_dir / "resolution.duckdb", args.cache_dir, args.threads)
    if args.import_records:
        import_split(connection, args.cache_dir, args.split)
    if args.index_targets:
        index_targets(connection, args.split)
    if args.candidates:
        sample = args.sample_per_thousand
        source_filter = f"hash(entity_id) % 1000 < {sample}" if sample else "TRUE"
        table = f"{args.split}_sample_candidates" if sample else f"{args.split}_candidates"
        generate_candidates(connection, args.split, table, source_filter, args.max_candidates)


if __name__ == "__main__":
    main()
