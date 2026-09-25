"""Report held-out link precision, recall, singleton accuracy, and macro F0.5."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

from .features import FEATURE_NAMES
from .train import dense_features, macro_f05, truth_for_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    metadata = json.loads((args.output_dir / "training_metrics.json").read_text())
    cap = int(metadata["candidate_cap"])
    threshold = float(metadata["threshold"])
    connection = duckdb.connect(str(args.database), read_only=True)
    validation_ids = {
        row[0] for row in connection.execute(
            "SELECT entity_id FROM train_source1 WHERE hash(entity_id) % 1000 BETWEEN 50 AND 59"
        ).fetchall()
    }
    truth = truth_for_ids(args.truth, validation_ids)
    fields = ", ".join(FEATURE_NAMES)
    data = connection.execute(f"""
        SELECT source1_entity_id, candidate_entity_id, {fields}
        FROM train_sample_features
        WHERE hash(source1_entity_id) % 1000 BETWEEN 50 AND 59
          AND candidate_rank <= {cap}
    """).fetchnumpy()
    sources = data["source1_entity_id"].tolist()
    targets = data["candidate_entity_id"].tolist()
    X = dense_features(data)
    model = xgb.Booster()
    model.load_model(args.output_dir / "pair_model.json")
    probabilities = model.predict(
        xgb.DMatrix(X, feature_names=list(FEATURE_NAMES)),
        iteration_range=(0, int(metadata["best_iteration"]) + 1),
    )
    predictions: dict[str, set[str]] = defaultdict(set)
    true_positives = false_positives = 0
    for source_id, target_id, probability in zip(sources, targets, probabilities):
        if probability >= threshold:
            predictions[source_id].add(target_id)
            if target_id in truth.get(source_id, set()):
                true_positives += 1
            else:
                false_positives += 1
    positives = sum(len(truth.get(source_id, set())) for source_id in validation_ids)
    false_negatives = positives - true_positives
    singletons = [source_id for source_id in validation_ids if not truth.get(source_id)]
    correct_singletons = sum(not predictions.get(source_id) for source_id in singletons)
    result = {
        "validation_entities": len(validation_ids),
        "true_positive_links": true_positives,
        "false_positive_links": false_positives,
        "false_negative_links": false_negatives,
        "link_precision": true_positives / (true_positives + false_positives),
        "link_recall": true_positives / positives,
        "true_singletons": len(singletons),
        "correct_singletons": correct_singletons,
        "singleton_accuracy": correct_singletons / len(singletons) if singletons else None,
        "macro_f05": macro_f05(
            sources, targets, probabilities, X[:, FEATURE_NAMES.index("candidate_rank")],
            truth, validation_ids, threshold, cap,
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
