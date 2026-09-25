"""Score test candidates and write both required submission TSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

from .blocking import connect, sql_string
from .features import FEATURE_NAMES, create_features


def write_grouped_output(
    connection: duckdb.DuckDBPyConnection, pairs_table: str,
    value_column: str, output_path: Path, output_column: str,
) -> None:
    assert pairs_table.replace("_", "").isalnum()
    assert value_column.replace("_", "").isalnum()
    assert output_column.replace("_", "").isalnum()
    connection.execute(f"""
        COPY (
            SELECT s.entity_id AS source1_entity_id,
                   string_agg(p.{value_column}, ',' ORDER BY p.{value_column})
                       AS {output_column}
            FROM test_source1 s
            LEFT JOIN {pairs_table} p ON p.source1_entity_id = s.entity_id
            GROUP BY s.entity_id
            ORDER BY s.entity_id
        ) TO '{sql_string(output_path)}' (HEADER true, DELIMITER '\\t')
    """)


def infer(database: Path, cache_dir: Path, output_dir: Path, batch_size: int = 100_000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((output_dir / "training_metrics.json").read_text())
    threshold = float(metadata["threshold"])
    model = xgb.Booster()
    model.load_model(output_dir / "pair_model.json")
    connection = connect(database, cache_dir)
    create_features(connection, "test", "test_candidates", "test_features")
    candidate_count = connection.execute("SELECT count(*) FROM test_candidates").fetchone()[0]
    feature_count = connection.execute("SELECT count(*) FROM test_features").fetchone()[0]
    if candidate_count != feature_count:
        raise RuntimeError(f"Candidate/feature mismatch: {candidate_count} != {feature_count}")

    print("Writing candidate_pairs.tsv", flush=True)
    write_grouped_output(
        connection, "test_features", "candidate_entity_id",
        output_dir / "candidate_pairs.tsv", "candidate_entity_ids",
    )
    fields = ", ".join(FEATURE_NAMES)
    cursor = connection.execute(
        f"SELECT source1_entity_id, candidate_entity_id, {fields} FROM test_features"
    )
    temporary = output_dir / "predicted_pairs.tmp.tsv"
    processed = selected = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("source1_entity_id", "candidate_entity_id"))
        while rows := cursor.fetchmany(batch_size):
            values = np.asarray([row[2:] for row in rows], dtype=np.float32)
            probabilities = model.predict(
                xgb.DMatrix(values, feature_names=list(FEATURE_NAMES)),
                iteration_range=(0, int(metadata["best_iteration"]) + 1),
            )
            for row, probability in zip(rows, probabilities):
                if probability >= threshold:
                    writer.writerow(row[:2])
                    selected += 1
            processed += len(rows)
            if processed % 1_000_000 < batch_size:
                print(f"Scored {processed:,}/{feature_count:,} candidates", flush=True)
    connection.execute(f"""
        CREATE OR REPLACE TABLE predicted_pairs AS
        SELECT * FROM read_csv('{sql_string(temporary)}', delim='\\t', header=true, all_varchar=true)
    """)
    print("Writing matching_results.tsv", flush=True)
    write_grouped_output(
        connection, "predicted_pairs", "candidate_entity_id",
        output_dir / "matching_results.tsv", "matched_entity_ids",
    )
    print(f"Scored {processed:,} candidates; selected {selected:,} matches", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    infer(args.database, args.cache_dir, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
