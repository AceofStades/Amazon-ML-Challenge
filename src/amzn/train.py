"""Train GPU XGBoost pair classifier and tune macro F0.5 on held-out entities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import xgboost as xgb

from .blocking import connect
from .features import FEATURE_NAMES, SECOND_STAGE_FEATURE_NAMES, create_features


def truth_for_ids(path: Path, ids: set[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source_id = row["source1_entity_id"]
            if source_id in ids:
                matches = row["matched_entity_ids"]
                result[source_id] = set(matches.split(",")) if matches else set()
    return result


def dense_features(data: dict, names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    columns = []
    for name in names:
        column = data[name]
        if np.ma.isMaskedArray(column):
            column = column.filled(np.nan)
        columns.append(np.asarray(column, dtype=np.float32))
    return np.column_stack(columns)


def macro_f05(
    source_ids: list[str], target_ids: list[str], probabilities: np.ndarray,
    candidate_ranks: np.ndarray, truth: dict[str, set[str]],
    validation_ids: set[str], threshold: float, candidate_cap: int,
) -> float:
    predicted: dict[str, set[str]] = defaultdict(set)
    for source_id, target_id, probability, rank in zip(
        source_ids, target_ids, probabilities, candidate_ranks
    ):
        if rank <= candidate_cap and probability >= threshold:
            predicted[source_id].add(target_id)
    total = 0.0
    for source_id in validation_ids:
        actual = truth.get(source_id, set())
        proposed = predicted.get(source_id, set())
        if not actual:
            total += float(not proposed)
        elif proposed:
            correct = len(actual & proposed)
            if correct:
                precision = correct / len(proposed)
                recall = correct / len(actual)
                total += 1.25 * precision * recall / (0.25 * precision + recall)
    return total / len(validation_ids)


def train(
    database: Path, cache_dir: Path, truth_path: Path, output_dir: Path,
    candidate_table: str = "train_sample_candidates",
    feature_table: str = "train_sample_features",
    validation_start: int = 50,
    validation_end: int = 60,
    rounds: int = 500,
    include_seed_features: bool = False,
    reuse_features: bool = False,
    max_depth: int = 8,
    eta: float = 0.07,
    min_child_weight: float = 5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = connect(database, cache_dir)
    names = SECOND_STAGE_FEATURE_NAMES if include_seed_features else FEATURE_NAMES
    if not reuse_features:
        create_features(connection, "train", candidate_table, feature_table,
                        include_seed_features=include_seed_features)
    fields = ", ".join(names)
    query = (
        f"SELECT source1_entity_id, candidate_entity_id, {fields}, "
        f"(hash(source1_entity_id) % 1000 BETWEEN {validation_start} "
        f"AND {validation_end - 1})::INT AS is_validation FROM {feature_table}"
    )
    data = connection.execute(query).fetchnumpy()
    source_ids = data["source1_entity_id"].tolist()
    target_ids = data["candidate_entity_id"].tolist()
    X = dense_features(data, names)
    validation_mask = np.asarray(data["is_validation"], dtype=bool)
    del data
    source_set = set(source_ids)
    validation_ids = {
        row[0] for row in connection.execute(
            f"SELECT entity_id FROM train_source1 WHERE hash(entity_id) % 1000 "
            f"BETWEEN {validation_start} AND {validation_end - 1}"
        ).fetchall()
    }
    source_set.update(validation_ids)
    truth = truth_for_ids(truth_path, source_set)
    empty: set[str] = set()
    labels = np.fromiter(
        (target_id in truth.get(source_id, empty)
         for source_id, target_id in zip(source_ids, target_ids)),
        dtype=np.float32, count=len(source_ids),
    )
    train_mask = ~validation_mask
    print(f"Training pairs: {train_mask.sum():,}; validation pairs: {validation_mask.sum():,}", flush=True)
    print(f"Training positives: {int(labels[train_mask].sum()):,}", flush=True)
    training = xgb.DMatrix(X[train_mask], label=labels[train_mask], feature_names=list(names))
    validation = xgb.DMatrix(X[validation_mask], label=labels[validation_mask], feature_names=list(names))
    model = xgb.train(
        {
            "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
            "device": "cuda", "max_depth": max_depth, "eta": eta, "subsample": 0.85,
            "colsample_bytree": 0.9, "min_child_weight": min_child_weight, "max_bin": 256,
            "nthread": 16, "seed": 42,
        },
        training, num_boost_round=rounds, evals=[(validation, "validation")],
        early_stopping_rounds=35, verbose_eval=25,
    )
    probabilities = model.predict(validation, iteration_range=(0, model.best_iteration + 1))
    validation_sources = [source_ids[index] for index in np.flatnonzero(validation_mask)]
    validation_targets = [target_ids[index] for index in np.flatnonzero(validation_mask)]
    validation_ranks = X[validation_mask, names.index("candidate_rank")]
    thresholds = np.arange(0.05, 0.951, 0.025)
    by_cap = {}
    maximum_rank = int(connection.execute(
        f"SELECT max(candidate_rank) FROM {feature_table}"
    ).fetchone()[0])
    caps = [cap for cap in (5, 10, 15, 20, 30, 40, 50, 60) if cap <= maximum_rank]
    for cap in caps:
        scored = [
            (float(threshold), macro_f05(
                validation_sources, validation_targets, probabilities, validation_ranks,
                truth, validation_ids, float(threshold), cap,
            ))
            for threshold in thresholds
        ]
        best_threshold, best_score = max(scored, key=lambda item: item[1])
        by_cap[cap] = {"threshold": best_threshold, "macro_f05": best_score}
        print(f"Candidate cap {cap}: macro F0.5={best_score:.6f} at {best_threshold:.3f}", flush=True)
    selected_cap = max(by_cap, key=lambda cap: by_cap[cap]["macro_f05"])
    threshold = by_cap[selected_cap]["threshold"]
    score = by_cap[selected_cap]["macro_f05"]
    model_path = output_dir / "pair_model.json"
    model.save_model(model_path)
    result = {
        "model": str(model_path), "features": list(names),
        "best_iteration": model.best_iteration, "threshold": threshold,
        "macro_f05": score, "candidate_cap": selected_cap,
        "validation_by_candidate_cap": by_cap,
        "validation_entities": len(validation_ids),
        "training_pairs": int(train_mask.sum()), "validation_pairs": int(validation_mask.sum()),
        "training_positives": int(labels[train_mask].sum()),
        "max_depth": max_depth, "eta": eta, "min_child_weight": min_child_weight,
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--candidate-table", default="train_sample_candidates")
    parser.add_argument("--feature-table", default="train_sample_features")
    parser.add_argument("--include-seed-features", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.07)
    parser.add_argument("--min-child-weight", type=float, default=5)
    args = parser.parse_args()
    train(args.database, args.cache_dir, args.truth, args.output_dir,
          candidate_table=args.candidate_table, feature_table=args.feature_table,
          rounds=args.rounds, include_seed_features=args.include_seed_features,
          reuse_features=args.reuse_features, max_depth=args.max_depth,
          eta=args.eta, min_child_weight=args.min_child_weight)


if __name__ == "__main__":
    main()
