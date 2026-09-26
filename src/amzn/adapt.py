"""Covariate-shift adapted stage-1 scorer for countries without labels.

The stage-1 matcher learns from US and Indian pairs, whose names are distinctive.
France (test only) has generic names and many look-alike records, so its pair
features follow another distribution. A domain classifier separates unlabelled
target-country pairs from training pairs; each training pair is weighted by the
density ratio p(target | x) / p(train | x), clipped, and a stage-1 model is
refitted with those weights. Only pairs that matter for decisions (stage-1
p >= 0.001) are used on both sides.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import xgboost as xgb

from .train_v3 import attach_labels
from .train_v4 import PAIR_HOUSE, address_numbers, pair_house_features
from .v3 import id_frames


def load_pairs(cache_dir: Path, model_dir: Path, split: str, names: list[str], keep: pl.DataFrame,
               numbers: dict) -> pl.DataFrame:
    stored = [n for n in names if n not in PAIR_HOUSE]
    parts = []
    for path in sorted(glob.glob(str(cache_dir / "v3" / split / "part*.parquet"))):
        part = pl.read_parquet(path, columns=["si", "ti", "bucket", "source1_entity_id", "candidate_entity_id",
                                              *stored])
        p1 = pl.read_parquet(model_dir / f"{split}_p1" / Path(path).name)
        part = part.join(p1, on=["si", "ti"]).filter(pl.col("p1") >= 0.001).join(keep, on="si", how="semi")
        parts.append(pair_house_features(part, numbers))
    return pl.concat(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--model-dir", type=Path, default=Path("output/v7"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--country", default="France")
    parser.add_argument("--output-dir", type=Path, default=Path("output/adapt_france"))
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = json.loads((args.model_dir / "stage1.json").read_text())["features"]

    train_ids, _ = id_frames(args.embed_dir, "train")
    buckets = duckdb.sql("SELECT si, (hash(entity_id) % 1000)::INT AS bucket FROM train_ids").pl()
    train_keep = buckets.filter(pl.col("bucket") < 100).select(pl.col("si").cast(pl.UInt32))
    test_ids = pl.read_parquet(args.embed_dir / "test_source1_ids.parquet").with_row_index("si")
    target_keep = test_ids.filter(pl.col("country") == args.country).select(pl.col("si").cast(pl.UInt32))

    train = load_pairs(args.cache_dir, args.model_dir, "train", names, train_keep,
                       address_numbers(args.cache_dir, "train"))
    train = attach_labels(train, args.truth)
    target = load_pairs(args.cache_dir, args.model_dir, "test", names, target_keep,
                        address_numbers(args.cache_dir, "test"))
    print(f"training pairs {train.height:,}; {args.country} test pairs {target.height:,}", flush=True)

    # Domain classifier on stage-1 features (labels: 1 = target country, 0 = training pair).
    X_train = train.select(names).cast(pl.Float32).to_numpy()
    X_target = target.select(names).cast(pl.Float32).to_numpy()
    rng = np.random.default_rng(0)
    take = rng.choice(len(X_train), size=min(len(X_train), len(X_target)), replace=False)
    D = xgb.DMatrix(np.vstack([X_train[take], X_target]),
                    label=np.r_[np.zeros(len(take)), np.ones(len(X_target))], feature_names=names)
    domain = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "tree_method": "hist",
                        "nthread": args.threads, "subsample": 0.8}, D, num_boost_round=300)
    ratio = len(X_target) / len(take)
    p = domain.predict(xgb.DMatrix(X_train, feature_names=names))
    weight = np.clip(p / np.clip(1 - p, 1e-6, None) / ratio, 1 / args.clip, args.clip)
    print("weights: mean %.3f, p50 %.3f, p90 %.3f, max %.3f" % (weight.mean(), *np.quantile(weight, [0.5, 0.9, 1.0])),
          flush=True)
    importance = sorted(domain.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:10]
    print("domain-shift features:", [k for k, _ in importance], flush=True)

    valid = ((train["bucket"] >= 50) & (train["bucket"] < 60)).to_numpy()
    labels = train["label"].to_numpy()
    dtrain = xgb.DMatrix(X_train[~valid], label=labels[~valid], weight=weight[~valid], feature_names=names)
    dvalid = xgb.DMatrix(X_train[valid], label=labels[valid], feature_names=names)
    model = xgb.train({"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 9, "eta": 0.08,
                       "subsample": 0.85, "colsample_bytree": 0.8, "min_child_weight": 3, "tree_method": "hist",
                       "nthread": args.threads, "seed": 5},
                      dtrain, num_boost_round=1500, evals=[(dvalid, "valid")], early_stopping_rounds=50,
                      verbose_eval=100)
    model.save_model(args.output_dir / "adapted_stage1.json")
    rounds = (0, model.best_iteration + 1)
    holdout = train.filter(pl.Series(valid)).select("si", "ti", "label").with_columns(
        p1=model.predict(dvalid, iteration_range=rounds).astype(np.float32))
    holdout = holdout.with_columns(pl.col("p1").alias("p2"))
    holdout.write_parquet(args.output_dir / "valid_predictions.parquet")
    scored = target.select("si", "ti").with_columns(
        p1=model.predict(xgb.DMatrix(X_target, feature_names=names), iteration_range=rounds).astype(np.float32))
    scored.with_columns(pl.col("p1").alias("p2"), pl.lit(0, pl.Int16).alias("bucket")).write_parquet(
        args.output_dir / "test_scored.parquet")
    print(f"wrote adapted scores for {scored.height:,} {args.country} pairs", flush=True)


if __name__ == "__main__":
    main()
