"""Measure blocking recall and candidate volume against training truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from .blocking import sql_string


def evaluate(database: Path, truth: Path, candidates: str, sample_per_thousand: int) -> dict:
    assert candidates.replace("_", "").isalnum()
    connection = duckdb.connect(str(database), read_only=True)
    scope = f"hash(source1_entity_id) % 1000 < {int(sample_per_thousand)}"
    recall = connection.execute(f"""
        WITH truth AS (
            SELECT source1_entity_id,
                   unnest(string_split(matched_entity_ids, ',')) AS target_id
            FROM read_csv('{sql_string(truth)}', delim='\\t', header=true, all_varchar=true)
            WHERE {scope} AND matched_entity_ids <> ''
        )
        SELECT count(*) AS positives,
               count(c.candidate_entity_id) AS found_30,
               count(*) FILTER (WHERE c.candidate_rank <= 5) AS found_5,
               count(*) FILTER (WHERE c.candidate_rank <= 10) AS found_10,
               count(*) FILTER (WHERE c.candidate_rank <= 15) AS found_15,
               count(*) FILTER (WHERE c.candidate_rank <= 20) AS found_20
        FROM truth t
        LEFT JOIN {candidates} c
          ON t.source1_entity_id = c.source1_entity_id
         AND t.target_id = c.candidate_entity_id
    """).fetchone()
    candidate_count = connection.execute(f"SELECT count(*) FROM {candidates}").fetchone()[0]
    source_count = connection.execute(
        f"SELECT count(*) FROM train_source1 WHERE hash(entity_id) % 1000 < {int(sample_per_thousand)}"
    ).fetchone()[0]
    result = {
        "source1_records": source_count,
        "candidate_pairs": candidate_count,
        "candidates_per_source1": candidate_count / source_count,
        "positive_pairs": recall[0],
    }
    for rank, found in zip((30, 5, 10, 15, 20), recall[1:]):
        result[f"recall_at_{rank}"] = found / recall[0] if recall[0] else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--candidates", default="train_sample_candidates")
    parser.add_argument("--sample-per-thousand", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.database, args.truth, args.candidates, args.sample_per_thousand), indent=2))


if __name__ == "__main__":
    main()
