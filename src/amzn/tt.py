"""Target-target same-entity model: separates decoy clusters from the true cluster.

Test data holds many decoy entities (absent from Source 1) whose records share
a perturbed name and a nearby house number with a Source 1 entity. A single
decoy record can look like a noisy true record; what differs is which other
candidates it agrees with. For each Source 1 entity, its top candidates by
stage-1 score are paired with each other and a classifier estimates whether two
target records describe the same business.

Labels (training split): both records matched to the same Source 1 entity -> 1;
matched to different entities, or exactly one matched (unmatched records belong
to entities absent from Source 1) -> 0; neither matched -> unlabelled. Training
uses only pairs drawn from Source 1 entities in hash buckets >= 100 that never
occur under a bucket < 100 entity, so stage-2 training entities stay unseen.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .blocking import connect
from .embed import target_vectors
from .features import create_features
from .v3 import id_frames

TT_DROP = {"block_weight", "matching_keys", "candidate_rank", "target_is_source3"}
AGGREGATES = ("tt_anchor", "tt_conf_mean", "tt_conf_max", "tt_low_max", "tt_mass", "tt_rank_in_s")


def top_pairs(model_dir: Path, split: str, top: int, floor: float) -> pl.DataFrame:
    """(si, ta, tb) for the top candidates of every Source 1 entity, ta < tb."""
    p1 = (pl.scan_parquet(model_dir / f"{split}_p1" / "*.parquet")
          .filter(pl.col("p1") >= floor)
          .with_columns(pl.col("p1").rank("ordinal", descending=True).over("si").alias("r"))
          .filter(pl.col("r") <= top).select("si", "ti", "p1").collect())
    left = p1.rename({"ti": "ta", "p1": "pa"})
    right = p1.rename({"ti": "tb", "p1": "pb"})
    pairs = left.join(right, on="si").filter(pl.col("ta") < pl.col("tb"))
    print(f"{split}: {p1.height:,} top candidates, {pairs.height:,} target pairs", flush=True)
    return pairs


def build(args: argparse.Namespace) -> None:
    out = args.cache_dir / "tt"
    out.mkdir(parents=True, exist_ok=True)
    pairs = top_pairs(args.model_dir, args.split, args.top, args.floor)
    pairs.write_parquet(out / f"{args.split}_pairs.parquet")
    unique = pairs.select("ta", "tb").unique()
    print(f"unique target pairs {unique.height:,}", flush=True)
    s_ids, t_ids = id_frames(args.embed_dir, args.split)
    named = (unique.join(t_ids.rename({"ti": "ta", "entity_id": "source1_entity_id"}), on="ta")
             .join(t_ids.rename({"ti": "tb", "entity_id": "candidate_entity_id"}), on="tb"))
    os.environ.setdefault("AMZN_DUCKDB_MEMORY", "8GB")
    connection = connect(args.database, args.cache_dir)
    connection.register("tt_frame", named.select("source1_entity_id", "candidate_entity_id").to_arrow())
    table = f"{args.split}_tt_candidates"
    connection.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT f.source1_entity_id, f.candidate_entity_id,
               0 AS block_weight, 0 AS matching_keys, 0 AS candidate_rank,
               jaro_winkler_similarity(a.name, b.name) AS name_jw,
               CASE WHEN a.address <> '' AND b.address <> ''
                    THEN jaro_winkler_similarity(a.address, b.address) ELSE 0 END AS address_jw
        FROM tt_frame f
        JOIN {args.split}_targets a ON f.source1_entity_id = a.entity_id
        JOIN {args.split}_targets b ON f.candidate_entity_id = b.entity_id
    """)
    features = f"{args.split}_tt_features"
    create_features(connection, args.split, table, features, left_table=f"{args.split}_targets")
    frame = connection.execute(f"SELECT * FROM {features}").pl()
    connection.execute(f"DROP TABLE {table}")
    connection.execute(f"DROP TABLE {features}")
    connection.close()
    frame = (frame.join(t_ids.rename({"ti": "ta", "entity_id": "source1_entity_id"}), on="source1_entity_id")
             .join(t_ids.rename({"ti": "tb", "entity_id": "candidate_entity_id"}), on="candidate_entity_id"))
    vectors = target_vectors(args.embed_dir, args.split)
    cosine = np.empty(frame.height, np.float32)
    ta, tb = frame["ta"].to_numpy(), frame["tb"].to_numpy()
    for start in range(0, frame.height, 1 << 20):
        end = start + (1 << 20)
        cosine[start:end] = np.einsum("nd,nd->n", vectors[ta[start:end]].astype(np.float32),
                                      vectors[tb[start:end]].astype(np.float32))
    frame = frame.with_columns(tt_cosine=cosine,
                               same_source=(pl.col("source1_entity_id").str.slice(0, 3)
                                            == pl.col("candidate_entity_id").str.slice(0, 3)).cast(pl.Float32))
    frame.drop("source1_entity_id", "candidate_entity_id").write_parquet(out / f"{args.split}_features.parquet")
    print(f"wrote {args.split} target-pair features: {frame.height:,}", flush=True)


def tt_feature_names(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"ta", "tb", "label"} and c not in TT_DROP]


def labels_for(pairs: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    s_ids, t_ids = id_frames(args.embed_dir, "train")
    truth = (pl.read_csv(args.truth, separator="\t", quote_char=None, infer_schema=False,
                         missing_utf8_is_empty_string=True)
             .with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .join(t_ids, left_on="matched_entity_ids", right_on="entity_id")
             .select("ti", pl.col("source1_entity_id").alias("owner")))
    return (pairs.join(truth.rename({"ti": "ta", "owner": "oa"}), on="ta", how="left")
            .join(truth.rename({"ti": "tb", "owner": "ob"}), on="tb", how="left")
            .with_columns(
                pl.when(pl.col("oa").is_null() & pl.col("ob").is_null()).then(None)
                .when(pl.col("oa") == pl.col("ob")).then(1).otherwise(0).alias("label"))
            .drop("oa", "ob"))


def train(args: argparse.Namespace) -> None:
    out = args.cache_dir / "tt"
    pairs = pl.read_parquet(out / "train_pairs.parquet")
    s_ids, _ = id_frames(args.embed_dir, "train")
    import duckdb
    buckets = duckdb.sql("SELECT si, hash(entity_id) % 1000 AS bucket FROM s_ids").pl()
    pairs = pairs.join(buckets, on="si")
    protected = pairs.filter(pl.col("bucket") < 100).select("ta", "tb").unique()
    usable = (pairs.filter(pl.col("bucket") >= 100)
              .group_by("ta", "tb").agg(pl.col("bucket").min())
              .join(protected, on=["ta", "tb"], how="anti"))
    frame = pl.read_parquet(out / "train_features.parquet").join(usable, on=["ta", "tb"])
    frame = labels_for(frame, args).filter(pl.col("label").is_not_null())
    names = tt_feature_names(frame.drop("bucket"))
    valid_mask = (frame["bucket"] < 110).to_numpy()
    print(f"target-pair training rows {frame.height:,}, positive rate {frame['label'].mean():.3f}", flush=True)

    def matrix(part: pl.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(part.select(names).cast(pl.Float32).to_numpy(), label=part["label"].to_numpy(),
                           feature_names=names)
    model = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": "cuda",
         "max_depth": 8, "eta": 0.08, "subsample": 0.85, "colsample_bytree": 0.8, "min_child_weight": 3,
         "seed": 7},
        matrix(frame.filter(pl.Series(~valid_mask))), num_boost_round=2000,
        evals=[(matrix(frame.filter(pl.Series(valid_mask))), "valid")], early_stopping_rounds=50,
        verbose_eval=200)
    model.save_model(out / "tt_model.json")
    (out / "tt_model_meta.json").write_text(json.dumps({"features": names}, indent=1))
    print(f"target-pair model best iteration {model.best_iteration}", flush=True)


def predict_and_aggregate(args: argparse.Namespace) -> None:
    out = args.cache_dir / "tt"
    names = json.loads((out / "tt_model_meta.json").read_text())["features"]
    model = xgb.Booster()
    model.load_model(out / "tt_model.json")
    model.set_param({"device": "cuda"})
    best = model.attr("best_iteration")
    features = pl.read_parquet(out / f"{args.split}_features.parquet")
    same = model.predict(xgb.DMatrix(features.select(names).cast(pl.Float32).to_numpy(), feature_names=names),
                         iteration_range=(0, int(best) + 1))
    probability = features.select("ta", "tb").with_columns(tt=same.astype(np.float32))
    pairs = pl.read_parquet(out / f"{args.split}_pairs.parquet").join(probability, on=["ta", "tb"])
    # Both orientations so each candidate sees all of its partners under the same entity.
    both = pl.concat([
        pairs.select("si", pl.col("ta").alias("ti"), pl.col("tb").alias("tj"), pl.col("pa").alias("p"),
                     pl.col("pb").alias("pj"), "tt"),
        pairs.select("si", pl.col("tb").alias("ti"), pl.col("ta").alias("tj"), pl.col("pb").alias("p"),
                     pl.col("pa").alias("pj"), "tt"),
    ])
    anchor = (both.group_by("si", "ti")
              .agg(pl.col("tt").sort_by("pj", descending=True).first().alias("tt_anchor"),
                   ((pl.col("pj") * pl.col("tt")).sum() / pl.col("pj").sum().clip(1e-6)).alias("tt_conf_mean"),
                   pl.col("tt").filter(pl.col("pj") >= 0.5).max().fill_null(-1.0).alias("tt_conf_max"),
                   pl.col("tt").filter(pl.col("pj") < 0.5).max().fill_null(-1.0).alias("tt_low_max"),
                   pl.col("tt").sum().alias("tt_mass"),
                   pl.col("p").first().alias("p")))
    anchor = anchor.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("si").cast(pl.Float32).alias("tt_rank_in_s")).drop("p")
    anchor.with_columns([pl.col(c).cast(pl.Float32) for c in AGGREGATES]).write_parquet(
        out / f"{args.split}_aggregates.parquet")
    print(f"wrote {args.split} target-pair aggregates: {anchor.height:,} candidate rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "train", "aggregate"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--model-dir", type=Path, default=Path("output/v4"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--database", type=Path, default=Path("cache/resolution.duckdb"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--floor", type=float, default=0.01)
    args = parser.parse_args()
    started = time.time()
    {"build": build, "train": train, "aggregate": predict_and_aggregate}[args.command](args)
    print(f"{args.command} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
