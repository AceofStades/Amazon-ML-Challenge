"""Score v3 test candidates and write matching_results.tsv and candidate_pairs.tsv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .train_v3 import decode, stage2_context


def grouped_tsv(source1: pl.Series, pairs: pl.DataFrame, column: str, path: Path) -> None:
    """One row per Source 1 entity; empty list when it has no pairs."""
    lists = pairs.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").unique().sort().str.join(",").alias(column))
    table = (pl.DataFrame({"source1_entity_id": source1})
             .join(lists, on="source1_entity_id", how="left")
             .with_columns(pl.col(column).fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path, separator="\t", quote_style="never")
    print(f"wrote {path}: {table.height:,} rows, {pairs.height:,} ids", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--model-dir", type=Path, default=Path("output/v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--one-to-one", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    metrics = json.loads((args.model_dir / "metrics.json").read_text())
    names1, names2 = metrics["features_stage1"], metrics["features_stage2"]
    mode, *rest = metrics["stage2"]["best"].split("_")
    threshold = float(rest[0])
    mass = float(rest[1]) if len(rest) > 1 else 0.0
    folds = []
    for path in sorted(args.model_dir.glob("stage1_fold*.json")):
        model = xgb.Booster()
        model.load_model(path)
        model.set_param({"device": "cuda"})
        folds.append(model)
    model2 = xgb.Booster()
    model2.load_model(args.model_dir / "stage2.json")
    model2.set_param({"device": "cuda"})
    t_vec = np.concatenate([np.load(args.embed_dir / f"test_source{i}.npy", mmap_mode="r") for i in (2, 3)])
    scored = []
    parts = sorted((args.cache_dir / "v3" / "test").glob("part*.parquet"))
    for n, path in enumerate(parts):
        part = pl.read_parquet(path).sort("si", "ti")
        X1 = xgb.DMatrix(part.select(names1).cast(pl.Float32).to_numpy(), feature_names=names1)
        p1 = sum(model.predict(X1) for model in folds) / len(folds)
        part = stage2_context(part, p1.astype(np.float32), t_vec)
        X2 = xgb.DMatrix(part.select(names2).cast(pl.Float32).to_numpy(), feature_names=names2)
        p2 = model2.predict(X2, iteration_range=(0, metrics["stage2_best_iteration"] + 1))
        scored.append(part.select("source1_entity_id", "candidate_entity_id", "si", "ti")
                      .with_columns(p2=p2))
        print(f"scored partition {n + 1}/{len(parts)}: {part.height:,} pairs", flush=True)
    frame = pl.concat(scored)
    print(f"test pairs {frame.height:,}; decode {mode} threshold {threshold} mass {mass}", flush=True)

    if args.one_to_one:
        # Each target belongs to at most one Source 1 entity: keep its best-scoring pair.
        best = frame.group_by("ti").agg(pl.col("p2").max().alias("best"))
        frame = frame.join(best, on="ti").with_columns(
            pl.when(pl.col("p2") < pl.col("best")).then(0.0).otherwise(pl.col("p2")).alias("p2"))
    chosen = decode(frame["si"].to_numpy(), frame["candidate_entity_id"].to_numpy(),
                    frame["p2"].to_numpy(), mode, threshold, mass)
    id_of = dict(zip(frame["si"].to_list(), frame["source1_entity_id"].to_list()))
    matches = pl.DataFrame(
        [(id_of[k], t) for k, v in chosen.items() for t in v],
        schema=["source1_entity_id", "candidate_entity_id"], orient="row")
    source1 = pl.read_csv("dataset/test/test_source1.tsv", separator="\t", quote_char=None,
                          infer_schema=False)["entity_id"]
    grouped_tsv(source1, frame.select("source1_entity_id", "candidate_entity_id"),
                "candidate_entity_ids", args.output_dir / "candidate_pairs.tsv")
    grouped_tsv(source1, matches, "matched_entity_ids", args.output_dir / "matching_results.tsv")


if __name__ == "__main__":
    main()
