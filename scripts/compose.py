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
# Full-competition target context (--extra): stage-1 scores of every candidate entity of the target, which are computed
# the same way on training (out-of-fold) and test, unlike the claims above that only see the scored entities.
EXTRA = ["q1", "m1", "top_other1", "s_tok_eq", "s_fold_eq", "n_s_tok_eq"]
EXTRA2 = ["o_tok_eq", "o_fold_eq"]
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


def features(frame: pl.DataFrame, info: pl.DataFrame, src: dict, na: dict, lo: float = LO, hi: float = HI) -> pl.DataFrame:
    """frame: si, ti, p (all candidate pairs of the entities). Returns ambiguous pairs with features."""
    x = frame.join(info, on="ti", how="left")
    conf = x.filter(pl.col("p") >= CONF)
    ent = conf.group_by("si").agg((1 - pl.col("t_s3")).sum().alias("C2"), pl.col("t_s3").sum().alias("C3"),
                                  pl.col("t_noaddr").sum().alias("CNA"))
    kinds = conf.filter(pl.col("t_kind") > 0).group_by("si", "t_kind").len().rename({"len": "same_kind_conf"})
    amb = x.filter((pl.col("p") >= lo) & (pl.col("p") < hi)).join(ent, on="si", how="left").fill_null(0)
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


def _fold(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def _tokset(s):
    import re
    out, run = [], ""
    for w in re.findall(r"[a-z0-9]+", _fold(s)):
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return " ".join(sorted(set(out)))


def extra_features(amb: pl.DataFrame, split: str) -> pl.DataFrame:
    """q1: the pair's stage-1 score over the sum for all candidate entities of the target (p1 >= 0.01); m1: their
    number; top_other1: best other entity's p1; raw token-set / folded-string equality of the names, and how many
    candidate entities of the target share the token set."""
    import glob
    targets = amb.select("ti").unique()
    p1 = pl.concat([pl.read_parquet(f).join(targets, on="ti", how="semi").filter(pl.col("p1") >= 0.01)
                    for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))])
    p1 = pl.concat([p1, amb.select("si", "ti").join(p1, on=["si", "ti"], how="anti").with_columns(pl.lit(0.01, dtype=pl.Float32).alias("p1"))])
    raw1 = read(f"dataset/{split}/{split}_source1.tsv", ["business_name"])["business_name"]
    raw2 = pl.concat([read(f"dataset/{split}/{split}_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
    s_tok = pl.DataFrame({"si": p1["si"].unique()})
    s_tok = s_tok.with_columns(raw1.gather(s_tok["si"].to_numpy()).alias("sn"))
    s_tok = s_tok.with_columns(pl.col("sn").map_elements(_tokset, return_dtype=pl.String).alias("s_tok"),
                               pl.col("sn").map_elements(_fold, return_dtype=pl.String).alias("s_fold")).drop("sn")
    t_tok = targets.with_columns(raw2.gather(targets["ti"].to_numpy()).alias("tn"))
    t_tok = t_tok.with_columns(pl.col("tn").map_elements(_tokset, return_dtype=pl.String).alias("t_tok"),
                               pl.col("tn").map_elements(_fold, return_dtype=pl.String).alias("t_fold")).drop("tn")
    c = p1.join(s_tok, on="si").join(t_tok, on="ti").with_columns(
        (pl.col("s_tok") == pl.col("t_tok")).cast(pl.Float32).alias("s_tok_eq"),
        (pl.col("s_fold") == pl.col("t_fold")).cast(pl.Float32).alias("s_fold_eq"))
    c = c.with_columns((pl.col("p1") / pl.col("p1").sum().over("ti")).alias("q1"), pl.len().over("ti").cast(pl.Float32).alias("m1"),
                       pl.col("s_tok_eq").sum().over("ti").alias("n_s_tok_eq"))
    best2 = c.group_by("ti").agg(pl.col("p1").sort(descending=True).head(2).alias("top"))
    c = c.join(best2, on="ti").with_columns(
        pl.when(pl.col("p1") >= pl.col("top").list.get(0)).then(pl.col("top").list.get(1, null_on_oob=True)).otherwise(pl.col("top").list.get(0))
        .fill_null(0.0).alias("top_other1"))
    return amb.join(c.select("si", "ti", *EXTRA), on=["si", "ti"], how="left").with_columns([pl.col(f).fill_null(0.0) for f in EXTRA])


def extra2_features(amb: pl.DataFrame, frame: pl.DataFrame, split: str) -> pl.DataFrame:
    """o_tok_eq / o_fold_eq: the target's raw token set / folded name equals that of one of the entity's confident
    records (p >= CONF), i.e. the same noisy variant already matched to this entity."""
    raw2 = pl.concat([read(f"dataset/{split}/{split}_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
    conf = frame.filter(pl.col("p") >= CONF).select("si", pl.col("ti").alias("ci")).join(amb.select("si").unique(), on="si", how="semi")
    need = pl.concat([conf.select(pl.col("ci").alias("ti")), amb.select("ti")]).unique()
    tok = need.with_columns(raw2.gather(need["ti"].to_numpy()).alias("n"))
    tok = tok.with_columns(pl.col("n").map_elements(_tokset, return_dtype=pl.String).alias("tok"),
                           pl.col("n").map_elements(_fold, return_dtype=pl.String).alias("fold")).drop("n")
    c = conf.join(tok.rename({"ti": "ci", "tok": "ctok", "fold": "cfold"}), on="ci")
    a = amb.select("si", "ti").join(tok, on="ti")
    m = a.join(c, on="si").filter(pl.col("ci") != pl.col("ti")).group_by("si", "ti").agg(
        (pl.col("tok") == pl.col("ctok")).any().cast(pl.Float32).alias("o_tok_eq"),
        (pl.col("fold") == pl.col("cfold")).any().cast(pl.Float32).alias("o_fold_eq"))
    return amb.join(m, on=["si", "ti"], how="left").with_columns([pl.col(f).fill_null(0.0) for f in EXTRA2])


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
    amb = features(scores.select("si", "ti", "p"), info, src, na, args.lo, args.hi).join(scores.select("si", "ti", "label", "bucket"), on=["si", "ti"])
    feats = FEATURES + (EXTRA if args.extra else []) + (EXTRA2 if args.extra2 else [])
    if args.extra:
        amb = extra_features(amb, "train")
    if args.extra2:
        amb = extra2_features(amb, scores.select("si", "ti", "p"), "train")
    print(f"ambiguous pairs {amb.height:,}, positive rate {amb['label'].mean():.3f}", flush=True)
    X, y = amb.select(feats).to_numpy(), amb["label"].to_numpy()
    fold = (amb["bucket"] // 10).to_numpy()
    p3 = np.zeros(amb.height, dtype=np.float32)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": args.depth, "eta": 0.1, "tree_method": "hist", "seed": 1}
    for k in range(10):
        fit, val = fold != k, fold == k
        model = xgb.train(params, xgb.DMatrix(X[fit], label=y[fit], feature_names=feats), 600,
                          evals=[(xgb.DMatrix(X[val], label=y[val], feature_names=feats), "v")], early_stopping_rounds=30,
                          verbose_eval=False)
        p3[val] = model.predict(xgb.DMatrix(X[val], feature_names=feats), iteration_range=(0, model.best_iteration + 1))
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
    # Test-like view: drop targets whose true owner is a training entity outside the cross-validated set (in test the
    # owner would be present and compete; here every claim on them is a false positive).
    owned = full.filter(pl.col("label") == 1).select("ti").unique()
    gt = (read("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "")
          .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids").select(pl.col("matched_entity_ids").alias("t")))
    tid = pl.concat([read(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    has_owner = tid.join(gt, left_on="entity_id", right_on="t", how="semi").select(pl.col("ti").cast(pl.UInt32))
    outside = has_owner.join(owned, on="ti", how="anti")
    kept = full.join(outside, on="ti", how="anti")
    b2, a2 = macro_f05(kept, "p", entities), macro_f05(kept, "p_new", entities)
    print(f"CV macro F0.5 without targets owned outside the CV set: {b2:.5f} -> {a2:.5f} ({a2 - b2:+.5f})", flush=True)
    if args.save:
        full.select("si", "ti", "label", "p", "p_new").write_parquet(args.save)
    imp = xgb.train(params, xgb.DMatrix(X, label=y, feature_names=feats), 300).get_score(importance_type="gain")
    print("gain:", sorted(((round(v, 1), k) for k, v in imp.items()), reverse=True)[:8], flush=True)


def test(args) -> None:
    from amzn.v3 import id_frames
    from amzn.infer_v3 import grouped_tsv
    scores = pl.read_parquet(args.cv_scores).with_columns(pl.col("p2").alias("p"))
    src, na = joint_tables()
    amb = features(scores.select("si", "ti", "p"), target_info("train", scores["ti"]), src, na, args.lo, args.hi).join(
        scores.select("si", "ti", "label"), on=["si", "ti"])
    feats = FEATURES + (EXTRA if args.extra else []) + (EXTRA2 if args.extra2 else [])
    if args.extra:
        amb = extra_features(amb, "train")
    if args.extra2:
        amb = extra2_features(amb, scores.select("si", "ti", "p"), "train")
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": args.depth, "eta": 0.1, "tree_method": "hist", "seed": 1}
    model = xgb.train(params, xgb.DMatrix(amb.select(feats).to_numpy(), label=amb["label"].to_numpy(), feature_names=feats),
                      args.rounds)
    countries = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").select("country").with_row_index("si")
    cal = pl.read_parquet(args.test_calibrated).join(countries, on="si").filter(pl.col("country").is_in(args.countries.split(",")))
    tamb = features(cal.select("si", "ti", "p"), target_info("test", cal["ti"]), src, na, args.lo, args.hi)
    if args.extra:
        tamb = extra_features(tamb, "test")
    if args.extra2:
        tamb = extra2_features(tamb, cal.select("si", "ti", "p"), "test")
    tamb = tamb.with_columns(pl.Series("p3", model.predict(xgb.DMatrix(tamb.select(feats).to_numpy(), feature_names=feats))))
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
    c.add_argument("--extra", action="store_true", help="add full-competition target context features")
    c.add_argument("--save", help="write the cross-validated scores to this parquet")
    t = sub.add_parser("test")
    t.add_argument("--cv-scores", required=True)
    t.add_argument("--test-calibrated", required=True)
    t.add_argument("--countries", default="US,India")
    t.add_argument("--rounds", type=int, default=250)
    t.add_argument("--output", required=True)
    t.add_argument("--extra", action="store_true", help="add full-competition target context features")
    for sp in (c, t):
        sp.add_argument("--extra2", action="store_true", help="add raw-name agreement with the entity's confident records")
        sp.add_argument("--lo", type=float, default=LO)
        sp.add_argument("--hi", type=float, default=HI)
        sp.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()
    if args.cmd == "cv":
        cv(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
