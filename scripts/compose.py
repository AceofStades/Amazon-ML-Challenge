"""Composition-aware rescoring of ambiguous pairs.

The generator spreads each entity's records over both sources (an entity with three records has none in Source 2
only 5.9% of the time, against 12.5% for a random split) and rarely gives an entity two records without an address.
A record without an address whose name is shared by several Source 1 entities is otherwise ambiguous, but it most
likely belongs to the entity whose confident records lack its source or lack an address-less record. For every
ambiguous pair this adds composition features computed from the Source 1 entity's confident matches (and from the
competing entities that also claim the target), and an XGBoost model rescores the pair.

  train (cross-validated over buckets 0-99): python scripts/compose.py cv --scores output/cv_auto2.parquet
  test:  python scripts/compose.py test --cv-scores output/cv_auto2.parquet --test-calibrated output/hybrid_X/test_calibrated.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

sys.path.insert(0, "src")
from amzn.train_v3 import decode  # noqa: E402
from amzn.train_v4 import one_to_one  # noqa: E402

CONF = 0.9
LO, HI = 0.01, 0.99
FEATURES = ["logit_p", "t_s3", "t_noaddr", "c2", "c3", "cna", "ctot", "fit_src", "fit_na", "n_claims", "other_max",
            "other_fit_src", "other_fit_na", "rel_fit", "rank_in_t", "t_kind", "same_kind_conf"]
KINDS = {"other": 0, "domain": 1, "acronym": 2, "alias": 3, "brand": 4}


def read(path, columns=None):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True,
                       columns=columns)


def record_kind(name: str | None) -> int:
    import re
    from amzn.france_rule import kind
    base = kind(name)
    if base != "other":
        return KINDS[base]
    words = re.findall(r"[A-Za-z]+", name or "")
    return KINDS["brand"] if len(words) == 1 and len(words[0]) >= 6 else 0


def target_info(split: str, needed: pl.Series | None = None) -> pl.DataFrame:
    s2 = read(f"dataset/{split}/{split}_source2.tsv", ["business_address"]).height
    rec = pl.concat([read(f"dataset/{split}/{split}_source{i}.tsv", ["business_name", "business_address"]) for i in (2, 3)])
    info = pl.DataFrame({"ti": np.arange(rec.height, dtype=np.uint32),
                         "t_s3": (np.arange(rec.height) >= s2).astype(np.float32),
                         "t_noaddr": (rec["business_address"] == "").to_numpy().astype(np.float32),
                         "name": rec["business_name"]})
    if needed is not None:
        info = info.join(pl.DataFrame({"ti": needed.unique()}), on="ti", how="semi")
    return info.with_columns(pl.col("name").map_elements(record_kind, return_dtype=pl.Int64).cast(pl.Float32).alias("t_kind")).drop("name")


def joint_tables() -> tuple[dict, dict]:
    g = (read("dataset/train/train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    t = pl.concat([read(f"dataset/train/train_source{i}.tsv", ["entity_id", "business_address"]) for i in (2, 3)])
    g = g.join(t, left_on="matched_entity_ids", right_on="entity_id", how="left").with_columns(
        pl.col("matched_entity_ids").str.starts_with("S3").alias("s3"), (pl.col("business_address") == "").alias("na"))
    per = g.group_by("source1_entity_id").agg((~pl.col("s3")).sum().alias("n2"), pl.col("s3").sum().alias("n3"),
                                              pl.col("na").sum().alias("k"))
    src = {(a, b): c for a, b, c in per.group_by("n2", "n3").len().rows()}
    na = {a: c for a, c in per.group_by("k").len().rows()}
    return src, na


def log_count(table: dict, key) -> float:
    return float(np.log(table.get(key, 0) + 0.5))


def features(frame: pl.DataFrame, info: pl.DataFrame, src: dict, na: dict) -> pl.DataFrame:
    """frame: si, ti, p (all candidate pairs of the entities). Returns ambiguous pairs with features."""
    x = frame.join(info, on="ti", how="left")
    conf = x.filter(pl.col("p") >= CONF)
    ent = conf.group_by("si").agg((1 - pl.col("t_s3")).sum().alias("C2"), pl.col("t_s3").sum().alias("C3"),
                                  pl.col("t_noaddr").sum().alias("CNA"))
    kinds = conf.filter(pl.col("t_kind") > 0).group_by("si", "t_kind").len().rename({"len": "same_kind_conf"})
    amb = x.filter((pl.col("p") >= LO) & (pl.col("p") < HI)).join(ent, on="si", how="left").fill_null(0)
    amb = amb.with_columns(pl.col("C2").alias("c2"), pl.col("C3").alias("c3"), pl.col("CNA").alias("cna"))
    amb = amb.join(kinds, on=["si", "t_kind"], how="left").with_columns(
        pl.when(pl.col("t_kind") > 0).then(pl.col("same_kind_conf").fill_null(0)).otherwise(0).cast(pl.Float32).alias("same_kind_conf"))
    c2, c3, cna = amb["c2"].to_numpy().astype(int), amb["c3"].to_numpy().astype(int), amb["cna"].to_numpy().astype(int)
    s3, tna = amb["t_s3"].to_numpy().astype(int), amb["t_noaddr"].to_numpy().astype(int)
    fit_src = np.array([log_count(src, (a + (1 - s), b + s)) - log_count(src, (a, b)) for a, b, s in zip(c2, c3, s3)])
    fit_na = np.array([log_count(na, k + t) - log_count(na, k) for k, t in zip(cna, tna)])
    amb = amb.with_columns(pl.Series("fit_src", fit_src.astype(np.float32)), pl.Series("fit_na", fit_na.astype(np.float32)),
                           (pl.col("c2") + pl.col("c3")).alias("ctot"),
                           (pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log().alias("logit_p"))
    # Competition for the target among Source 1 entities.
    claims = amb.select("si", "ti", "p", "fit_src", "fit_na")
    per_t = claims.group_by("ti").agg(pl.len().alias("n_claims"))
    amb = amb.join(per_t, on="ti", how="left")
    other = claims.rename({"si": "si_o", "p": "p_o", "fit_src": "fs_o", "fit_na": "fn_o"})
    comp = (amb.select("si", "ti").join(other, on="ti").filter(pl.col("si_o") != pl.col("si"))
            .group_by("si", "ti").agg(pl.col("p_o").max().alias("other_max"), pl.col("fs_o").max().alias("other_fit_src"),
                                      pl.col("fn_o").max().alias("other_fit_na")))
    amb = amb.join(comp, on=["si", "ti"], how="left").with_columns(
        pl.col("other_max").fill_null(0.0), pl.col("other_fit_src").fill_null(-5.0), pl.col("other_fit_na").fill_null(-5.0))
    amb = amb.with_columns((pl.col("fit_src") + pl.col("fit_na") - pl.max_horizontal("other_fit_src", pl.lit(-5.0))
                            - pl.max_horizontal("other_fit_na", pl.lit(-5.0))).alias("rel_fit"),
                           pl.col("p").rank("ordinal", descending=True).over("ti").cast(pl.Float32).alias("rank_in_t"))
    return amb


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


def cv(args) -> None:
    scores = pl.read_parquet(args.scores).with_columns(pl.col("p2").alias("p"))
    src, na = joint_tables()
    info = target_info("train", scores["ti"])
    amb = features(scores.select("si", "ti", "p"), info, src, na).join(scores.select("si", "ti", "label", "bucket"), on=["si", "ti"])
    print(f"ambiguous pairs {amb.height:,}, positive rate {amb['label'].mean():.3f}", flush=True)
    X, y = amb.select(FEATURES).to_numpy(), amb["label"].to_numpy()
    fold = (amb["bucket"] // 10).to_numpy()
    p3 = np.zeros(amb.height, dtype=np.float32)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5, "eta": 0.1, "tree_method": "hist", "seed": 1}
    for k in range(10):
        fit, val = fold != k, fold == k
        model = xgb.train(params, xgb.DMatrix(X[fit], label=y[fit], feature_names=FEATURES), 600,
                          evals=[(xgb.DMatrix(X[val], label=y[val], feature_names=FEATURES), "v")], early_stopping_rounds=30,
                          verbose_eval=False)
        p3[val] = model.predict(xgb.DMatrix(X[val], feature_names=FEATURES), iteration_range=(0, model.best_iteration + 1))
    amb = amb.with_columns(pl.Series("p3", p3))
    ll2 = -np.mean(y * np.log(np.clip(amb["p"].to_numpy(), 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - amb["p"].to_numpy(), 1e-6, 1)))
    ll3 = -np.mean(y * np.log(np.clip(p3, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p3, 1e-6, 1)))
    print(f"ambiguous-pair logloss: blend {ll2:.4f} -> composition {ll3:.4f}", flush=True)
    full = scores.join(amb.select("si", "ti", "p3"), on=["si", "ti"], how="left").with_columns(
        pl.coalesce("p3", "p").alias("p_new"))
    entities = scores["si"].n_unique()
    before = macro_f05(full, "p", entities)
    after = macro_f05(full, "p_new", entities)
    print(f"CV macro F0.5 over {entities:,} entities: {before:.5f} -> {after:.5f} ({after - before:+.5f})", flush=True)
    imp = xgb.train(params, xgb.DMatrix(X, label=y, feature_names=FEATURES), 300).get_score(importance_type="gain")
    print("gain:", sorted(((round(v, 1), k) for k, v in imp.items()), reverse=True)[:8], flush=True)


def test(args) -> None:
    from amzn.v3 import id_frames
    from amzn.infer_v3 import grouped_tsv
    scores = pl.read_parquet(args.cv_scores).with_columns(pl.col("p2").alias("p"))
    src, na = joint_tables()
    amb = features(scores.select("si", "ti", "p"), target_info("train", scores["ti"]), src, na).join(
        scores.select("si", "ti", "label"), on=["si", "ti"])
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 5, "eta": 0.1, "tree_method": "hist", "seed": 1}
    model = xgb.train(params, xgb.DMatrix(amb.select(FEATURES).to_numpy(), label=amb["label"].to_numpy(), feature_names=FEATURES),
                      args.rounds)
    countries = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").select("country").with_row_index("si")
    cal = pl.read_parquet(args.test_calibrated).join(countries, on="si").filter(pl.col("country").is_in(args.countries.split(",")))
    tamb = features(cal.select("si", "ti", "p"), target_info("test", cal["ti"]), src, na)
    tamb = tamb.with_columns(pl.Series("p3", model.predict(xgb.DMatrix(tamb.select(FEATURES).to_numpy(), feature_names=FEATURES))))
    print(f"test ambiguous pairs {tamb.height:,}; mean p {tamb['p'].mean():.4f} -> {tamb['p3'].mean():.4f}", flush=True)
    full = cal.join(tamb.select("si", "ti", "p3"), on=["si", "ti"], how="left").with_columns(pl.coalesce("p3", "p").alias("p_new"))
    scored = one_to_one(full.select("si", "ti", "p_new", "raw"), "p_new", tie_break="raw")
    chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p_new"].to_numpy(), "expected", 0.05, 0.0)
    matches = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                            "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
        pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
    s_ids, t_ids = id_frames(Path("cache/embeddings"), "test")
    named = (matches.join(s_ids.rename({"entity_id": "source1_entity_id"}), on="si")
             .join(t_ids.rename({"entity_id": "candidate_entity_id"}), on="ti"))
    grouped_tsv(s_ids["entity_id"], named, "matched_entity_ids", Path(args.output))
    print(f"wrote {args.output}: {matches.height:,} matches for {args.countries}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cv")
    c.add_argument("--scores", required=True)
    t = sub.add_parser("test")
    t.add_argument("--cv-scores", required=True)
    t.add_argument("--test-calibrated", required=True)
    t.add_argument("--countries", default="US,India")
    t.add_argument("--rounds", type=int, default=250)
    t.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cmd == "cv":
        cv(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
