"""Score first-pass candidates to select precise sibling-expansion seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

from .blocking import connect, sql_string


def create_seeds(
    connection: duckdb.DuckDBPyConnection,
    feature_table: str,
    model_dir: Path,
    output_table: str,
    cache_dir: Path,
    threshold: float = 0.9,
    batch_size: int = 100_000,
) -> None:
    for table in (feature_table, output_table):
        assert table.replace("_", "").isalnum()
    metadata = json.loads((model_dir / "training_metrics.json").read_text())
    names = tuple(metadata["features"])
    model = xgb.Booster()
    model.load_model(model_dir / "pair_model.json")
    model.set_param({"device": "cuda"})
    cursor = connection.execute(
        f"SELECT source1_entity_id, candidate_entity_id, {', '.join(names)} FROM {feature_table}"
    )
    path = cache_dir / f"{output_table}.tsv"
    count = selected = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("source1_entity_id", "seed_id"))
        while rows := cursor.fetchmany(batch_size):
            values = np.asarray([row[2:] for row in rows], dtype=np.float32)
            probabilities = model.predict(
                xgb.DMatrix(values, feature_names=list(names)),
                iteration_range=(0, int(metadata["best_iteration"]) + 1),
            )
            for row, probability in zip(rows, probabilities):
                if probability >= threshold:
                    writer.writerow(row[:2])
                    selected += 1
            count += len(rows)
            if count % 1_000_000 < batch_size:
                print(f"Seeded {count:,} pairs; {selected:,} selected", flush=True)
    connection.execute(f"""
        CREATE OR REPLACE TABLE {output_table} AS
        SELECT * FROM read_csv('{sql_string(path)}', delim='\\t', header=true, all_varchar=true)
    """)
    print(f"{output_table}: {selected:,} seeds from {count:,} pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--feature-table", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()
    connection = connect(args.database, args.cache_dir)
    create_seeds(connection, args.feature_table, args.model_dir, args.output_table,
                 args.cache_dir, args.threshold)


if __name__ == "__main__":
    main()
