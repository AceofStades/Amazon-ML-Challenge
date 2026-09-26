"""Disk-backed, bounded blocking with several independent match keys."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import duckdb


def connect(path: Path, cache_dir: Path, threads: int = 16) -> duckdb.DuckDBPyConnection:
    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(path))
    connection.execute(f"SET threads = {int(threads)}")
    connection.execute(f"SET memory_limit = '{os.environ.get('AMZN_DUCKDB_MEMORY', '14GB')}'")
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


def expand_candidates(
    connection: duckdb.DuckDBPyConnection,
    split: str,
    seeds_table: str,
    base_table: str,
    output_table: str,
    max_candidates: int = 30,
    max_key_frequency: int = 40,
    base_cap: int = 30,
    source_filter: str = "TRUE",
) -> None:
    """Retrieve near neighbors of confident target matches as extra candidates."""
    assert split in {"train", "test"}
    for table in (seeds_table, base_table, output_table):
        assert table.replace("_", "").isalnum()
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS {split}_target_key_frequencies AS
        SELECT key, count(*) AS frequency FROM {split}_target_keys GROUP BY key
    """)
    connection.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        WITH seed_keys AS (
            SELECT DISTINCT s.source1_entity_id, s.seed_id, k.key
            FROM {seeds_table} s
            JOIN {split}_target_keys k ON s.seed_id = k.entity_id
            JOIN {split}_target_key_frequencies f ON k.key = f.key
            WHERE f.frequency <= {int(max_key_frequency)} AND ({source_filter})
        ),
        hits AS (
            SELECT sk.source1_entity_id, sk.seed_id,
                   tk.entity_id AS candidate_entity_id,
                   count(*) AS matching_keys
            FROM seed_keys sk
            JOIN {split}_target_keys tk USING (key)
            WHERE sk.seed_id <> tk.entity_id
            GROUP BY 1, 2, 3
        ),
        scored AS (
            SELECT h.source1_entity_id, h.candidate_entity_id, h.matching_keys,
                   jaro_winkler_similarity(seed.name, target.name) AS seed_name_jw,
                   coalesce(jaro_winkler_similarity(seed.address, target.address), 0) AS seed_address_jw,
                   70 * jaro_winkler_similarity(seed.name, target.name)
                   + 60 * coalesce(jaro_winkler_similarity(seed.address, target.address), 0)
                   + 3 * h.matching_keys AS seed_score,
                   jaro_winkler_similarity(source.name, target.name) AS name_jw,
                   coalesce(jaro_winkler_similarity(source.address, target.address), 0) AS address_jw,
                   (source.house <> '' AND source.house = target.house)::INT AS house_equal
            FROM hits h
            JOIN {split}_source1 source ON h.source1_entity_id = source.entity_id
            JOIN {split}_targets seed ON h.seed_id = seed.entity_id
            JOIN {split}_targets target ON h.candidate_entity_id = target.entity_id
            ANTI JOIN {base_table} b ON h.source1_entity_id = b.source1_entity_id
                                    AND h.candidate_entity_id = b.candidate_entity_id
        ),
        grouped AS (
            SELECT source1_entity_id, candidate_entity_id,
                   max(matching_keys) AS matching_keys,
                   max(seed_score) AS seed_score,
                   max(seed_name_jw) AS seed_name_jw,
                   max(seed_address_jw) AS seed_address_jw,
                   max(name_jw) AS name_jw,
                   max(address_jw) AS address_jw,
                   max(house_equal) AS house_equal
            FROM scored GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY source1_entity_id
                ORDER BY seed_score + 70 * name_jw + 60 * address_jw
                         + 12 * house_equal DESC, candidate_entity_id
            ) AS expansion_rank
            FROM grouped
        )
        SELECT source1_entity_id, candidate_entity_id,
               0 AS block_weight, matching_keys,
               name_jw, address_jw, house_equal,
               {int(base_cap)} + expansion_rank AS candidate_rank,
               seed_score, seed_name_jw, seed_address_jw
        FROM ranked WHERE expansion_rank <= {int(max_candidates)}
    """)
    count, entities = connection.execute(
        f"SELECT count(*), count(DISTINCT source1_entity_id) FROM {output_table}"
    ).fetchone()
    print(f"{output_table}: {count:,} sibling candidates for {entities:,} Source 1 records", flush=True)


def combine_candidates(
    connection: duckdb.DuckDBPyConnection, base_table: str,
    expanded_table: str, output_table: str,
) -> None:
    """Keep exactly the union that the final classifier will score."""
    for table in (base_table, expanded_table, output_table):
        assert table.replace("_", "").isalnum()
    fields = (
        "source1_entity_id, candidate_entity_id, block_weight, matching_keys, "
        "name_jw, address_jw, house_equal, candidate_rank"
    )
    connection.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT {fields}, 0.0 AS seed_score, 0.0 AS seed_name_jw,
               0.0 AS seed_address_jw FROM {base_table}
        UNION ALL
        SELECT {fields}, seed_score, seed_name_jw, seed_address_jw
        FROM {expanded_table}
    """)
    print(f"{output_table}: {connection.execute(f'SELECT count(*) FROM {output_table}').fetchone()[0]:,} pairs", flush=True)


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
    parser.add_argument("--partitions", type=int, default=1,
                        help="Partition Source 1 joins to bound temporary disk use")
    parser.add_argument("--partition-index", type=int)
    parser.add_argument("--start-partition", type=int, default=0)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--expanded-table")
    parser.add_argument("--seeds-table")
    parser.add_argument("--base-table")
    parser.add_argument("--output-table")
    parser.add_argument("--max-key-frequency", type=int, default=40)
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
        if args.partitions < 1:
            parser.error("--partitions must be positive")
        if args.partitions == 1:
            generate_candidates(connection, args.split, table, source_filter, args.max_candidates)
        else:
            staging = f"{table}_staging"
            if args.partition_index is None:
                connection.close()
                for part in range(args.start_partition, args.partitions):
                    subprocess.run([
                        sys.executable, "-m", "amzn.blocking", "--split", args.split,
                        "--candidates", "--sample-per-thousand", str(sample),
                        "--max-candidates", str(args.max_candidates),
                        "--cache-dir", str(args.cache_dir),
                        "--threads", str(args.threads),
                        "--partitions", str(args.partitions),
                        "--partition-index", str(part),
                    ], check=True)
                connection = connect(args.cache_dir / "resolution.duckdb", args.cache_dir,
                                     args.threads)
                connection.execute(f"DROP TABLE IF EXISTS {table}")
                connection.execute(f"ALTER TABLE {staging} RENAME TO {table}")
                print(f"{table}: {connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]:,} total pairs", flush=True)
            else:
                part = args.partition_index
                if not 0 <= part < args.partitions:
                    parser.error("--partition-index is outside --partitions")
                part_table = f"{table}_part"
                part_filter = f"({source_filter}) AND hash(entity_id) % {args.partitions} = {part}"
                print(f"Candidate partition {part + 1}/{args.partitions}", flush=True)
                generate_candidates(connection, args.split, part_table, part_filter,
                                    args.max_candidates)
                if part == 0:
                    connection.execute(f"CREATE OR REPLACE TABLE {staging} AS SELECT * FROM {part_table}")
                else:
                    connection.execute(f"INSERT INTO {staging} SELECT * FROM {part_table}")
                connection.execute(f"DROP TABLE {part_table}")
    if args.expand:
        if not all((args.seeds_table, args.base_table, args.output_table)):
            parser.error("--expand requires --seeds-table, --base-table, and --output-table")
        if args.partitions < 1:
            parser.error("--partitions must be positive")
        if args.partitions == 1:
            expand_candidates(connection, args.split, args.seeds_table, args.base_table,
                              args.output_table, args.max_candidates, args.max_key_frequency)
        else:
            staging = f"{args.output_table}_staging"
            if args.partition_index is None:
                connection.close()
                for part in range(args.start_partition, args.partitions):
                    subprocess.run([
                        sys.executable, "-m", "amzn.blocking", "--split", args.split,
                        "--expand", "--seeds-table", args.seeds_table,
                        "--base-table", args.base_table,
                        "--output-table", args.output_table,
                        "--max-candidates", str(args.max_candidates),
                        "--max-key-frequency", str(args.max_key_frequency),
                        "--cache-dir", str(args.cache_dir),
                        "--threads", str(args.threads),
                        "--partitions", str(args.partitions),
                        "--partition-index", str(part),
                    ], check=True)
                connection = connect(args.cache_dir / "resolution.duckdb", args.cache_dir,
                                     args.threads)
                connection.execute(f"DROP TABLE IF EXISTS {args.output_table}")
                connection.execute(f"ALTER TABLE {staging} RENAME TO {args.output_table}")
                print(f"{args.output_table}: {connection.execute(f'SELECT count(*) FROM {args.output_table}').fetchone()[0]:,} total sibling pairs", flush=True)
            else:
                part = args.partition_index
                if not 0 <= part < args.partitions:
                    parser.error("--partition-index is outside --partitions")
                part_table = f"{args.output_table}_part"
                print(f"Expansion partition {part + 1}/{args.partitions}", flush=True)
                expand_candidates(connection, args.split, args.seeds_table, args.base_table,
                                  part_table, args.max_candidates, args.max_key_frequency,
                                  source_filter=f"hash(s.source1_entity_id) % {args.partitions} = {part}")
                if part == 0:
                    connection.execute(f"CREATE OR REPLACE TABLE {staging} AS SELECT * FROM {part_table}")
                else:
                    connection.execute(f"INSERT INTO {staging} SELECT * FROM {part_table}")
                connection.execute(f"DROP TABLE {part_table}")
    if args.combine:
        if not all((args.base_table, args.expanded_table, args.output_table)):
            parser.error("--combine requires --base-table, --expanded-table, and --output-table")
        combine_candidates(connection, args.base_table, args.expanded_table,
                           args.output_table)


if __name__ == "__main__":
    main()
