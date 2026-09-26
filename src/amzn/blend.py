"""Pairwise blend of stage-1 scores with cross-encoder logits (no entity context).

Stage-2 context features amplify the test decoys; purely pairwise signals transfer
better. This fits a shallow XGBoost on (logit p1, cross-encoder logit) using
training entities in buckets < 100 outside the holdout, where p1 is out-of-fold
and the cross-encoder never saw the entity. Pairs without a cross-encoder score
(p1 < 0.01) keep p1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import xgboost as xgb

from .v3 import id_frames

BASE = ["logit_p1", "ce_logit"]
HOUSE = ["house_equal", "house_conflict"]
TT = ["tt_anchor", "tt_conf_mean", "tt_conf_max", "tt_low_max", "tt_mass", "tt_rank_in_s"]
FULL: list[str] = []
NAME = ["s_name_dup", "t_name_dup", "s_name_in_targets", "name_idf_cos", "name_idf_max_shared",
        "address_idf_cos", "house_match", "house_conflict"]


FEATURES = BASE


def with_features(frame: pl.DataFrame) -> pl.DataFrame:
    p = pl.col("p1").clip(1e-6, 1 - 1e-6)
    return frame.with_columns((p / (1 - p)).log().alias("logit_p1"))


def load(model_dir: Path, ce_dir: Path, split: str, extras: set[str], cache_dir: Path = Path("cache"),
         second: Path | None = None, extra_ce: tuple[tuple[str, Path], ...] = (),
         keep: pl.DataFrame | None = None) -> pl.DataFrame:
    p1 = pl.scan_parquet(model_dir / f"{split}_p1" / "*.parquet")
    if keep is not None:
        p1 = p1.join(keep.lazy(), on="si", how="semi")
    ce = pl.scan_parquet(ce_dir / f"{split}_scores.parquet").with_columns(
        pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
    frame = p1.join(ce, on=["si", "ti"], how="left").collect()
    for name, folder in extra_ce:
        other = pl.read_parquet(folder / f"{split}_scores.parquet").select(
            pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.col("ce_logit").alias(f"{name}_logit"))
        frame = frame.join(other, on=["si", "ti"], how="left")
    if second is not None:
        other = pl.scan_parquet(second / f"{split}_p1" / "*.parquet").rename({"p1": "p1b"}).collect()
        p = pl.col("p1b").clip(1e-6, 1 - 1e-6)
        frame = frame.join(other, on=["si", "ti"], how="left").with_columns((p / (1 - p)).log().alias("logit_p1b"))
    if "full" in extras:
        import glob
        names = [n for n in json.loads((model_dir / "stage1.json").read_text())["features"]]
        live = frame.filter(pl.col("ce_logit").is_not_null()).select("si", "ti")
        first = sorted(glob.glob(str(cache_dir / "v3" / split / "part*.parquet")))
        available = [n for n in names if n in pl.read_parquet_schema(first[0])]
        parts = [pl.read_parquet(path, columns=["si", "ti", *available]).join(live, on=["si", "ti"], how="semi")
                 for path in first]
        full = pl.concat(parts).rename({n: f"f_{n}" for n in available})
        frame = frame.join(full, on=["si", "ti"], how="left")
        FULL.extend(f"f_{n}" for n in available if f"f_{n}" not in FULL)
    if "name" in extras:
        import glob
        live = frame.filter(pl.col("ce_logit").is_not_null()).select("si", "ti")
        parts = [pl.read_parquet(path, columns=["si", "ti", *NAME]).join(live, on=["si", "ti"], how="semi")
                 for path in sorted(glob.glob(str(cache_dir / "v3" / split / "part*.parquet")))]
        frame = frame.join(pl.concat(parts), on=["si", "ti"], how="left")
    if "tt" in extras:
        tt = pl.read_parquet(cache_dir / "tt" / f"{split}_aggregates.parquet").with_columns(
            pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
        frame = frame.join(tt, on=["si", "ti"], how="left")
    if "house" in extras:
        from .train_v4 import house_numbers
        s_house, _, t_numbers = house_numbers(cache_dir, split)
        live = frame.filter(pl.col("ce_logit").is_not_null()).select("si", "ti")
        hs = pl.Series(s_house[live["si"].to_numpy()])
        nums = t_numbers.gather(live["ti"].to_numpy())
        relation = pl.DataFrame({"hs": hs, "nums": nums}).select(
            ((pl.col("hs") != "") & pl.col("nums").list.contains(pl.col("hs"))).cast(pl.Float32).alias("house_equal"),
            ((pl.col("hs") != "") & (pl.col("nums").list.len() > 0)
             & ~pl.col("nums").list.contains(pl.col("hs"))).cast(pl.Float32).alias("house_conflict"))
        frame = frame.join(pl.concat([live, relation], how="horizontal"), on=["si", "ti"], how="left")
    return with_features(frame)


def labels(frame: pl.DataFrame, embed_dir: Path, truth: Path) -> pl.DataFrame:
    s_ids, t_ids = id_frames(embed_dir, "train")
    positives = (pl.read_csv(truth, separator="\t", quote_char=None, infer_schema=False,
                             missing_utf8_is_empty_string=True)
                 .with_columns(pl.col("matched_entity_ids").str.split(","))
                 .explode("matched_entity_ids")
                 .join(s_ids, left_on="source1_entity_id", right_on="entity_id")
                 .join(t_ids, left_on="matched_entity_ids", right_on="entity_id")
                 .select("si", "ti", pl.lit(1, pl.Int8).alias("label")))
    return frame.join(positives, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("output/v4"))
    parser.add_argument("--ce-dir", type=Path, default=Path("cache/ce"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/v4_blend"))
    parser.add_argument("--extras", default="", help="comma list: house, tt, name, full")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--test-chunks", type=int, default=8)
    parser.add_argument("--second-model-dir", type=Path, help="Another stage-1 run to blend in")
    parser.add_argument("--extra-ce", action="append", default=[],
                        help="NAME=DIR with DIR/{split}_scores.parquet from another cross-encoder")
    args = parser.parse_args()
    extras = {e for e in args.extras.split(",") if e}
    global FEATURES
    FEATURES = (BASE + (HOUSE if "house" in extras else []) + (TT if "tt" in extras else [])
                + (NAME if "name" in extras else []) + (["logit_p1b"] if args.second_model_dir else []))
    extra_ce = tuple((item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.extra_ce)
    FEATURES += [f"{name}_logit" for name, _ in extra_ce]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    s_ids, _ = id_frames(args.embed_dir, "train")
    buckets = duckdb.sql("SELECT si, (hash(entity_id) % 1000)::INT AS bucket FROM s_ids").pl()
    keep = buckets.filter(pl.col("bucket") < 100).select(pl.col("si").cast(pl.UInt32))
    train = load(args.model_dir, args.ce_dir, "train", extras, second=args.second_model_dir, extra_ce=extra_ce,
                 keep=keep)
    FEATURES += [n for n in FULL if n not in FEATURES]
    train = train.join(buckets.with_columns(pl.col("si").cast(pl.UInt32)), on="si").filter(pl.col("bucket") < 100)
    train = labels(train, args.embed_dir, args.truth)
    valid_mask = (pl.col("bucket") >= 50) & (pl.col("bucket") < 60)
    scored = train.filter(pl.col("ce_logit").is_not_null())
    fit_rows, valid_rows = scored.filter(~valid_mask), scored.filter(valid_mask)
    print(f"blend rows: fit {fit_rows.height:,}, holdout {valid_rows.height:,}", flush=True)

    def matrix(frame: pl.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(frame.select(FEATURES).to_numpy(), label=frame["label"].to_numpy(), feature_names=FEATURES)
    model = xgb.train({"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": args.depth, "eta": 0.1,
                       "tree_method": "hist", "seed": 3},
                      matrix(fit_rows), num_boost_round=1000, evals=[(matrix(valid_rows), "valid")],
                      early_stopping_rounds=30, verbose_eval=100)
    model.save_model(args.output_dir / "blend.json")
    (args.output_dir / "blend_meta.json").write_text(json.dumps({"features": FEATURES}))
    best = model.best_iteration

    def blend(frame: pl.DataFrame) -> pl.DataFrame:
        has = frame["ce_logit"].is_not_null().to_numpy()
        p = frame["p1"].to_numpy().astype(np.float32).copy()
        if has.any():
            X = frame.filter(pl.Series(has)).select(FEATURES).to_numpy()
            p[has] = model.predict(xgb.DMatrix(X, feature_names=FEATURES), iteration_range=(0, best + 1))
        return frame.with_columns(p3=p)

    holdout = blend(train.filter(valid_mask))
    holdout.select("si", "ti", "p1", "p3", "label").rename({"p3": "p2"}).write_parquet(
        args.output_dir / "valid_predictions.parquet")
    test_ids, _ = id_frames(args.embed_dir, "test")
    chunks = []
    for chunk in range(args.test_chunks):
        keep = test_ids.filter(pl.col("si") % args.test_chunks == chunk).select(pl.col("si").cast(pl.UInt32))
        part = blend(load(args.model_dir, args.ce_dir, "test", extras, second=args.second_model_dir,
                          extra_ce=extra_ce, keep=keep))
        chunks.append(part.select("si", "ti", "p1", pl.col("p3").alias("p2")))
        del part
    pl.concat(chunks).with_columns(pl.lit(0, pl.Int16).alias("bucket")).write_parquet(
        args.output_dir / "test_scored.parquet")
    (args.output_dir / "evaluation.json").write_text(json.dumps({"best": "p2_o2o_expected_0.05_0.0"}))
    print("wrote blend predictions", flush=True)


if __name__ == "__main__":
    main()
