"""10-fold cross-validated blend evaluation over every training entity in buckets 0-99 (about 220k entities).

The single holdout (buckets 50-59, 22k entities) cannot separate blends that differ by 0.0002 in F0.5. Here each
fold of ten buckets is predicted by a blend fitted on the other ninety, and macro F0.5 is computed over all
out-of-fold entities with the hybrid's decoding. Usage mirrors amzn.blend:
  python scripts/blend_cv.py --extras name,full,gen --extra-ce b=cache/ce_b ... --tag NAME
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, "src")
from amzn import blend as B  # noqa: E402
from amzn.train_v3 import decode  # noqa: E402
from amzn.train_v4 import one_to_one  # noqa: E402
from amzn.v3 import id_frames  # noqa: E402


def macro_f05(frame: pl.DataFrame, column: str, entities: int) -> float:
    scored = one_to_one(frame.select("si", "ti", column), column)
    chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored[column].to_numpy(), "expected", 0.05, 0.0)
    picked = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                           "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
        pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32), pl.lit(True).alias("c"))
    h = frame.select("si", "ti", "label").join(picked, on=["si", "ti"], how="left").with_columns(pl.col("c").fill_null(False))
    g = h.group_by("si").agg(pl.col("label").sum().alias("a"), pl.col("c").sum().alias("k"),
                             (pl.col("label").cast(pl.Boolean) & pl.col("c")).sum().alias("tp"))
    a, k, tp = g["a"].to_numpy(), g["k"].to_numpy(), g["tp"].to_numpy()
    f = np.where((a == 0) & (k == 0), 1.0, np.where((a == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * a + k, 1e-9)))
    return float((f.sum() + entities - len(g)) / entities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("output/v7"))
    parser.add_argument("--ce-dir", type=Path, default=Path("cache/ce"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--extras", default="name,full,gen")
    parser.add_argument("--extra-ce", action="append", default=[])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    extras = {e for e in args.extras.split(",") if e}
    extra_ce = tuple((item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.extra_ce)
    features = (B.BASE + (B.NAME if "name" in extras else []) + (B.GEN if "gen" in extras else [])
                + (B.CTX if "ctx" in extras else []) + (B.CTX2 if "ctx2" in extras else [])
                + [f"{name}_logit" for name, _ in extra_ce])
    s_ids, _ = id_frames(args.embed_dir, "train")
    buckets = duckdb.sql("SELECT si, (hash(entity_id) % 1000)::INT AS bucket FROM s_ids").pl().with_columns(pl.col("si").cast(pl.UInt32))
    keep = buckets.filter(pl.col("bucket") < 100).select("si")
    train = B.load(args.model_dir, args.ce_dir, "train", extras, extra_ce=extra_ce, keep=keep)
    features += [n for n in B.FULL if n not in features]
    train = B.labels(train.join(buckets, on="si"), args.embed_dir, args.truth)
    has = train["ce_logit"].is_not_null()
    out = train.select("si", "ti", "label", "bucket", "p1").with_columns(pl.col("p1").alias("p2"))
    p2 = out["p2"].to_numpy().copy()
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": args.depth, "eta": args.eta,
              "tree_method": "hist", "seed": 3}
    fold = (train["bucket"] // 10).to_numpy()
    X = train.select(features).to_numpy()
    y = train["label"].to_numpy()
    mask = has.to_numpy()
    losses = []
    for k in range(10):
        valid = mask & (fold == k)
        fit = mask & (fold != k)
        dfit = xgb.DMatrix(X[fit], label=y[fit], feature_names=features)
        dvalid = xgb.DMatrix(X[valid], label=y[valid], feature_names=features)
        model = xgb.train(params, dfit, num_boost_round=3000, evals=[(dvalid, "valid")], early_stopping_rounds=30,
                          verbose_eval=False)
        p2[valid] = model.predict(dvalid, iteration_range=(0, model.best_iteration + 1))
        losses.append(model.best_score)
        print(f"  fold {k}: logloss {model.best_score:.5f}, rounds {model.best_iteration + 1}", flush=True)
    out = out.with_columns(pl.Series("p2", p2.astype(np.float32)))
    out.write_parquet(f"output/cv_{args.tag}.parquet")
    entities = keep.height
    f05 = macro_f05(out, "p2", entities)
    per_fold = [macro_f05(out.filter(pl.col("bucket") // 10 == k), "p2",
                          buckets.filter(pl.col("bucket") // 10 == k).height) for k in range(10)]
    print(f"CV {args.tag}: F0.5 {f05:.5f} over {entities:,} entities; mean fold logloss {np.mean(losses):.5f}; "
          f"fold F0.5 {' '.join(f'{v:.4f}' for v in per_fold)}", flush=True)


if __name__ == "__main__":
    main()
