"""Full-split two-stage matcher with target competition features.

Every Source 1 entity of a split has candidates and features (cache/v3/{split}),
so stage-1 scores exist for all pairs. For each target, stage 2 then sees how
strongly other Source 1 entities claim it. The same code path scores the test
split, and the holdout evaluation applies the one-target-one-entity rule over
all training entities, exactly as at test time.

Buckets are hash(source1_entity_id) % 1000. Training entities: buckets below
--train-buckets except the holdout 50-59. The encoder was fine-tuned on
buckets >= 100, so those entities only act as competitors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

from .embed import target_vectors
from .train_v3 import (STAGE2_EXTRA, attach_labels, decode, feature_names, macro_f05,
                       stage2_context, truth_sets)
from .v3 import id_frames

COMPETITION = ("t_p1_other_max", "t_p1_other_second", "t_claims50", "p1_minus_t_other",
               "p1_share_t", "t_candidates")
HOUSE = ("t_house_known", "t_house_is_s", "t_house_size", "t_house_other_p1", "t_house_cluster_rank",
         "s_house_size", "s_house_max_p1", "best_other_house_p1", "p1_minus_s_house_max",
         "house_log_gap", "house_digit_diffs", "house_first_diff", "house_last_diff", "house_same_length")


def house_numbers(cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, pl.Series]:
    """First house number per record (si and ti order, '' when absent) and, per target,
    every number in its address (leading zeros removed)."""
    def read(i: int, columns: list[str]) -> pl.DataFrame:
        return pl.read_csv(cache_dir / f"{split}_source{i}.tsv", separator="\t", quote_char=None,
                           infer_schema=False, columns=columns)
    source1 = read(1, ["house"])["house"].fill_null("")
    targets = pl.concat([read(2, ["house", "address"]), read(3, ["house", "address"])])
    numbers = targets["address"].fill_null("").str.extract_all(r"\d+").list.eval(
        pl.element().str.strip_chars_start("0").replace("", "0")).alias("numbers")
    return source1.to_numpy(), targets["house"].fill_null("").to_numpy(), numbers


def house_context(part: pl.DataFrame, s_house: np.ndarray, t_house: np.ndarray,
                  t_numbers: pl.Series | None = None) -> pl.DataFrame:
    """Decoy entities sit a few doors away and bring several records sharing their own
    house number; a typo on a true record leaves its house number alone in the group.
    A target counts as at the Source 1 house when that number appears anywhere in its
    address (noise such as 'Door No #834 House No 1958' puts another number first)."""
    frame = part.select("si", "ti", "p1").with_columns(
        pl.Series("hs", s_house[part["si"].to_numpy()]), pl.Series("ht_first", t_house[part["ti"].to_numpy()]))
    if t_numbers is not None:
        frame = frame.with_columns(pl.Series("t_numbers", t_numbers.gather(part["ti"].to_numpy())))
        s_in_t = (pl.col("hs") != "") & pl.col("t_numbers").list.contains(pl.col("hs"))
    else:
        s_in_t = (pl.col("hs") != "") & (pl.col("hs") == pl.col("ht_first"))
    frame = frame.with_columns(pl.when(s_in_t).then(pl.col("hs")).otherwise(pl.col("ht_first")).alias("ht"))
    known = pl.col("ht") != ""
    clusters = (frame.filter(known).group_by("si", "ht")
                .agg(pl.len().alias("size"), pl.col("p1").max().alias("cmax"),
                     pl.col("p1").top_k(2).min().alias("csecond"))
                .with_columns(pl.col("cmax").rank("ordinal", descending=True).over("si").alias("crank")))
    s_side = (frame.filter(known & (pl.col("ht") == pl.col("hs"))).group_by("si")
              .agg(pl.len().alias("s_house_size"), pl.col("p1").max().alias("s_house_max_p1")))
    other = (clusters.join(frame.select("si", "hs").unique("si"), on="si")
             .filter(pl.col("ht") != pl.col("hs")).group_by("si")
             .agg(pl.col("cmax").max().alias("best_other_house_p1")))
    out = (frame.join(clusters, on=["si", "ht"], how="left").join(s_side, on="si", how="left")
           .join(other, on="si", how="left")
           .with_columns(
               known.cast(pl.Float32).alias("t_house_known"),
               (known & (pl.col("ht") == pl.col("hs"))).cast(pl.Float32).alias("t_house_is_s"),
               pl.col("size").fill_null(0).cast(pl.Float32).alias("t_house_size"),
               # Strongest other record in t's own house cluster.
               pl.when(pl.col("size") > 1).then(
                   pl.when(pl.col("p1") >= pl.col("cmax")).then(pl.col("csecond")).otherwise(pl.col("cmax")))
               .otherwise(-1.0).alias("t_house_other_p1"),
               pl.col("crank").fill_null(0).cast(pl.Float32).alias("t_house_cluster_rank"),
               pl.col("s_house_size").fill_null(0).cast(pl.Float32),
               pl.col("s_house_max_p1").fill_null(-1.0),
               pl.col("best_other_house_p1").fill_null(-1.0),
           )
           .with_columns((pl.col("p1") - pl.col("s_house_max_p1")).alias("p1_minus_s_house_max")))
    out = out.with_columns(house_shape(out["hs"], out["ht"]))
    return part.with_columns(out.select(HOUSE))


def house_shape(hs: pl.Series, ht: pl.Series) -> list[pl.Series]:
    """How two house numbers differ: decoys sit a few doors away (last digits change
    by a small offset); typos swap one digit anywhere or drop digits."""
    a, b = hs.to_numpy(), ht.to_numpy()
    n = len(a)
    gap = np.full(n, -1.0, np.float32)
    diffs = np.full(n, -1.0, np.float32)
    first = np.full(n, -1.0, np.float32)
    last = np.full(n, -1.0, np.float32)
    same_length = np.full(n, -1.0, np.float32)
    for i in range(n):
        x, y = a[i], b[i]
        if not x or not y:
            continue
        if len(x) < 10 and len(y) < 10:
            gap[i] = np.log1p(abs(int(x) - int(y)))
        same_length[i] = float(len(x) == len(y))
        if len(x) == len(y):
            positions = [k for k in range(len(x)) if x[k] != y[k]]
            diffs[i] = len(positions)
            if positions:
                first[i] = positions[0] / len(x)
                last[i] = (len(x) - 1 - positions[-1])
            else:
                first[i] = last[i] = 99.0
    return [pl.Series("house_log_gap", gap), pl.Series("house_digit_diffs", diffs),
            pl.Series("house_first_diff", first), pl.Series("house_last_diff", last),
            pl.Series("house_same_length", same_length)]
VALID = (50, 60)


def partitions(cache_dir: Path, split: str) -> list[Path]:
    return sorted((cache_dir / "v3" / split).glob("part*.parquet"))


def xgb_params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "device": "cuda", "max_depth": args.max_depth, "eta": args.eta, "subsample": 0.85,
        "colsample_bytree": 0.8, "min_child_weight": 3, "max_bin": 256, "seed": 42,
    }


def dmatrix(frame: pl.DataFrame, names: list[str], label: bool = True) -> xgb.DMatrix:
    X = frame.select(names).cast(pl.Float32).to_numpy()
    return xgb.DMatrix(X, label=frame["label"].to_numpy() if label else None, feature_names=names)


def is_valid(frame: pl.DataFrame) -> pl.Expr:
    return (pl.col("bucket") >= VALID[0]) & (pl.col("bucket") < VALID[1])


PAIR_HOUSE = ("s_house_in_t", "t_house_in_s", "house_min_gap", "closest_digit_diffs",
              "closest_first_diff", "closest_last_diff", "closest_same_length")


def address_numbers(cache_dir: Path, split: str) -> dict:
    """First house number and all address numbers for Source 1 (si) and targets (ti)."""
    def read(i: int) -> pl.DataFrame:
        return pl.read_csv(cache_dir / f"{split}_source{i}.tsv", separator="\t", quote_char=None,
                           infer_schema=False, columns=["house", "address"])
    def numbers(frame: pl.DataFrame) -> pl.Series:
        return frame["address"].fill_null("").str.extract_all(r"\d+").list.eval(
            pl.element().str.strip_chars_start("0").replace("", "0"))
    source1 = read(1)
    targets = pl.concat([read(2), read(3)])
    return {"s_house": source1["house"].fill_null("").to_numpy(), "s_numbers": numbers(source1),
            "t_house": targets["house"].fill_null("").to_numpy(), "t_numbers": numbers(targets)}


def pair_house_features(part: pl.DataFrame, numbers: dict) -> pl.DataFrame:
    """Pairwise house evidence robust to noise prefixes such as 'Door No #834 House No 1958':
    does either first house number appear anywhere in the other address, and how does the
    Source 1 house number differ from the closest target number."""
    si, ti = part["si"].to_numpy(), part["ti"].to_numpy()
    frame = pl.DataFrame({
        "row": np.arange(part.height), "hs": numbers["s_house"][si], "ht": numbers["t_house"][ti],
        "s_nums": numbers["s_numbers"].gather(si), "t_nums": numbers["t_numbers"].gather(ti),
    })
    frame = frame.with_columns(
        ((pl.col("hs") != "") & pl.col("t_nums").list.contains(pl.col("hs"))).cast(pl.Float32).alias("s_house_in_t"),
        ((pl.col("ht") != "") & pl.col("s_nums").list.contains(pl.col("ht"))).cast(pl.Float32).alias("t_house_in_s"))
    exploded = (frame.filter((pl.col("hs") != "") & (pl.col("t_nums").list.len() > 0))
                .select("row", "hs", "t_nums").explode("t_nums")
                .filter(pl.col("t_nums").str.len_chars() < 10)
                .with_columns((pl.col("hs").cast(pl.Int64, strict=False)
                               - pl.col("t_nums").cast(pl.Int64, strict=False)).abs().alias("gap"))
                .sort("gap").group_by("row").agg(pl.col("gap").first(), pl.col("t_nums").first().alias("closest")))
    frame = frame.join(exploded, on="row", how="left").sort("row")
    shape = house_shape(frame["hs"], frame["closest"].fill_null(""))
    return part.with_columns(
        frame["s_house_in_t"], frame["t_house_in_s"],
        pl.Series("house_min_gap", np.where(frame["gap"].is_null().to_numpy(), -1.0,
                                            np.log1p(frame["gap"].fill_null(0).cast(pl.Float64).to_numpy()))
                  .astype(np.float32)),
        shape[1].alias("closest_digit_diffs"), shape[2].alias("closest_first_diff"),
        shape[3].alias("closest_last_diff"), shape[4].alias("closest_same_length"))


def load_training(args: argparse.Namespace) -> pl.DataFrame:
    numbers = address_numbers(args.cache_dir, "train") if args.pair_house else None
    frames = []
    for p in partitions(args.cache_dir, "train"):
        part = pl.read_parquet(p).filter(pl.col("bucket") < args.train_buckets)
        if numbers is not None:
            part = pair_house_features(part, numbers)
        frames.append(part)
    return attach_labels(pl.concat(frames), args.truth).sort("si", "ti")


def fit(train: pl.DataFrame, valid: pl.DataFrame, names: list[str], args) -> xgb.Booster:
    return xgb.train(xgb_params(args), dmatrix(train, names), num_boost_round=args.rounds,
                     evals=[(dmatrix(valid, names), "valid")], early_stopping_rounds=50,
                     verbose_eval=200)


def predict(model: xgb.Booster, X: xgb.DMatrix) -> np.ndarray:
    best = model.attr("best_iteration")
    rounds = int(best) + 1 if best is not None else model.num_boosted_rounds()
    return model.predict(X, iteration_range=(0, rounds))


def train_stage1(args: argparse.Namespace) -> None:
    frame = load_training(args)
    names = feature_names(frame)
    valid = frame.filter(is_valid(frame))
    train = frame.filter(~is_valid(frame))
    print(f"stage1 train rows {train.height:,} valid rows {valid.height:,} features {len(names)}", flush=True)
    for fold in range(args.folds):
        model = fit(train.filter(pl.col("bucket") % args.folds != fold), valid, names, args)
        model.save_model(args.model_dir / f"stage1_fold{fold}.json")
        print(f"stage1 fold {fold}: best iteration {model.best_iteration}", flush=True)
    (args.model_dir / "stage1.json").write_text(json.dumps({"features": names}, indent=1))


def load_models(model_dir: Path, pattern: str) -> list[xgb.Booster]:
    models = []
    for path in sorted(model_dir.glob(pattern)):
        model = xgb.Booster()
        model.load_model(path)
        model.set_param({"device": "cuda"})
        models.append(model)
    return models


def score_stage1(args: argparse.Namespace, split: str) -> None:
    """p1 for every pair: out-of-fold for training buckets, fold average otherwise."""
    names = json.loads((args.model_dir / "stage1.json").read_text())["features"]
    models = load_models(args.model_dir, "stage1_fold*.json")
    out = args.model_dir / f"{split}_p1"
    out.mkdir(exist_ok=True)
    numbers = address_numbers(args.cache_dir, split) if any(n in PAIR_HOUSE for n in names) else None
    for path in partitions(args.cache_dir, split):
        part = pl.read_parquet(path, columns=["si", "ti", "bucket", *[n for n in names if n not in PAIR_HOUSE]])
        if numbers is not None:
            part = pair_house_features(part, numbers)
        X = xgb.DMatrix(part.select(names).cast(pl.Float32).to_numpy(), feature_names=names)
        scores = np.stack([predict(m, X) for m in models])
        p1 = scores.mean(0)
        if split == "train":
            bucket = part["bucket"].to_numpy()
            own = (bucket < args.train_buckets) & ~((bucket >= VALID[0]) & (bucket < VALID[1]))
            fold = bucket % args.folds
            p1 = np.where(own, scores[fold, np.arange(len(p1))], p1)
        part.select("si", "ti").with_columns(p1=p1.astype(np.float32)).write_parquet(out / path.name)
        print(f"{split} p1 {path.name}: {part.height:,}", flush=True)


def competition(args: argparse.Namespace, split: str) -> pl.DataFrame:
    """Per-target top-2 stage-1 scores across all Source 1 entities."""
    p1 = pl.scan_parquet(args.model_dir / f"{split}_p1" / "*.parquet")
    return p1.group_by("ti").agg(
        pl.col("p1").top_k(2).alias("top"), (pl.col("p1") > 0.5).sum().cast(pl.Float32).alias("claims50"),
        pl.col("p1").sum().alias("p1_sum_t"), pl.len().cast(pl.Float32).alias("t_candidates"),
    ).with_columns(
        pl.col("top").list.get(0, null_on_oob=True).alias("t_top1"),
        pl.col("top").list.get(1, null_on_oob=True).fill_null(0.0).alias("t_top2"),
    ).drop("top").collect()


def add_competition(part: pl.DataFrame, comp: pl.DataFrame) -> pl.DataFrame:
    part = part.join(comp, on="ti", how="left")
    mine_is_top = pl.col("p1") >= pl.col("t_top1")
    other_max = pl.when(mine_is_top).then(pl.col("t_top2")).otherwise(pl.col("t_top1"))
    return part.with_columns(
        other_max.alias("t_p1_other_max"),
        pl.when(mine_is_top).then(0.0).otherwise(pl.col("t_top2")).alias("t_p1_other_second"),
        (pl.col("claims50") - (pl.col("p1") > 0.5).cast(pl.Float32)).alias("t_claims50"),
        (pl.col("p1") - other_max).alias("p1_minus_t_other"),
        (pl.col("p1") / pl.col("p1_sum_t").clip(1e-6)).alias("p1_share_t"),
    ).drop("t_top1", "t_top2", "claims50", "p1_sum_t")


def extra_frames(templates: list[str], split: str) -> list[pl.LazyFrame]:
    """Extra per-pair feature files keyed by (si, ti); '{split}' is substituted."""
    return [pl.scan_parquet(t.format(split=split)) for t in templates]


def extra_columns(templates: list[str], split: str) -> list[str]:
    names: list[str] = []
    for frame in extra_frames(templates, split):
        names += [c for c in frame.collect_schema().names() if c not in ("si", "ti")]
    return names


def stage2_frame(args, split: str, path: Path, comp: pl.DataFrame, t_vec: np.ndarray,
                 houses: tuple[np.ndarray, np.ndarray], labels: bool,
                 max_bucket: int = 1000, extras: list[str] = ()) -> pl.DataFrame:
    part = pl.read_parquet(path).filter(pl.col("bucket") < max_bucket)
    p1 = pl.read_parquet(args.model_dir / f"{split}_p1" / path.name)
    part = part.join(p1, on=["si", "ti"]).sort("si", "ti")
    if labels:
        part = attach_labels(part, args.truth).sort("si", "ti")
    part = stage2_context(part.drop("p1"), part["p1"].to_numpy(), t_vec)
    part = house_context(part, *houses)
    part = add_competition(part, comp)
    if extras:
        sources = part.select(pl.col("si").unique()).lazy()
        for frame in extra_frames(list(extras), split):
            extra = frame.join(sources, on="si", how="semi").collect()
            extra = extra.with_columns(pl.col("si").cast(part.schema["si"]), pl.col("ti").cast(part.schema["ti"]))
            part = part.join(extra, on=["si", "ti"], how="left")
        part = part.sort("si", "ti")
    return part


def train_stage2(args: argparse.Namespace) -> None:
    comp = competition(args, "train")
    t_vec = target_vectors(args.embed_dir, "train")
    houses = house_numbers(args.cache_dir, "train")
    frames = []
    for path in partitions(args.cache_dir, "train"):
        frames.append(stage2_frame(args, "train", path, comp, t_vec, houses, labels=True,
                                   max_bucket=args.train_buckets, extras=args.extra))
    frame = pl.concat(frames)
    names1 = json.loads((args.model_dir / "stage1.json").read_text())["features"]
    names2 = (names1 + list(STAGE2_EXTRA) + list(COMPETITION) + list(HOUSE)
              + extra_columns(args.extra, "train"))
    valid = frame.filter(is_valid(frame))
    train = frame.filter(~is_valid(frame))
    print(f"stage2 train rows {train.height:,} valid rows {valid.height:,} features {len(names2)}", flush=True)
    model = fit(train, valid, names2, args)
    model.save_model(args.stage2_dir / "stage2.json")
    (args.stage2_dir / "stage2_meta.json").write_text(
        json.dumps({"features": names2, "extras": list(args.extra)}, indent=1))


def score_stage2(args: argparse.Namespace, split: str) -> pl.DataFrame:
    comp = competition(args, split)
    meta = json.loads((args.stage2_dir / "stage2_meta.json").read_text())
    model = load_models(args.stage2_dir, "stage2.json")[0]
    t_vec = target_vectors(args.embed_dir, split)
    houses = house_numbers(args.cache_dir, split)
    scored = []
    for path in partitions(args.cache_dir, split):
        part = stage2_frame(args, split, path, comp, t_vec, houses, labels=False,
                            extras=meta.get("extras", []))
        X = xgb.DMatrix(part.select(meta["features"]).cast(pl.Float32).to_numpy(), feature_names=meta["features"])
        scored.append(part.select("si", "ti", "bucket", "p1").with_columns(p2=predict(model, X)))
        print(f"{split} p2 {path.name}: {part.height:,}", flush=True)
    frame = pl.concat(scored)
    frame.write_parquet(args.stage2_dir / f"{split}_scored.parquet")
    return frame


def one_to_one(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    best = frame.group_by("ti").agg(pl.col(column).max().alias("best"))
    return frame.join(best, on="ti").with_columns(
        pl.when(pl.col(column) < pl.col("best")).then(0.0).otherwise(pl.col(column)).alias(column)
    ).drop("best")


def evaluate(args: argparse.Namespace) -> None:
    frame = pl.read_parquet(args.stage2_dir / "train_scored.parquet")
    s_ids, t_ids = id_frames(args.embed_dir, "train")
    valid_si = frame.filter(is_valid(frame))["si"].unique()
    all_valid = s_ids.join(pl.DataFrame({"si": valid_si}), on="si")
    import duckdb
    source1 = s_ids.select("entity_id")
    valid_ids = set(duckdb.sql(
        f"SELECT entity_id FROM source1 WHERE hash(entity_id) % 1000 BETWEEN {VALID[0]} AND {VALID[1] - 1}"
    ).pl()["entity_id"].to_list())
    truth = truth_sets(args.truth, valid_ids)
    id_of = dict(zip(all_valid["si"].to_list(), all_valid["entity_id"].to_list()))
    t_name = t_ids["entity_id"].to_numpy()
    results = {}
    for column in ("p1", "p2"):
        for o2o in (False, True):
            scored = one_to_one(frame, column) if o2o else frame
            scored = scored.filter(is_valid(scored))
            si, ti, p = scored["si"].to_numpy(), scored["ti"].to_numpy(), scored[column].to_numpy()
            for mode, th, mass in (("threshold", 0.5, 0), ("threshold", 0.6, 0), ("threshold", 0.7, 0),
                                   ("expected", 0.05, 0.0), ("expected", 0.05, 0.2)):
                chosen = decode(si, ti, p, mode, th, mass)
                named = {k: [t_name[t] for t in v] for k, v in chosen.items()}
                key = f"{column}_{'o2o' if o2o else 'free'}_{mode}_{th}_{mass}"
                results[key] = macro_f05(named, truth, id_of, valid_ids)
                print(key, round(results[key], 5), flush=True)
    best = max(results, key=results.get)
    (args.stage2_dir / "evaluation.json").write_text(json.dumps({"best": best, "all": results}, indent=1))
    print("best", best, results[best])
    # Labelled holdout pairs for test-time label-shift calibration.
    valid = frame.filter(is_valid(frame))
    positives = pl.DataFrame(
        [(s, t) for s, ts in truth.items() for t in ts], schema=["source1_entity_id", "candidate_entity_id"],
        orient="row")
    positives = (positives.join(s_ids.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
                 .join(t_ids.rename({"entity_id": "candidate_entity_id"}), on="candidate_entity_id")
                 .select("si", "ti", pl.lit(1, pl.Int8).alias("label")))
    valid = valid.join(positives, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
    valid.write_parquet(args.stage2_dir / "valid_predictions.parquet")


def write_test(args: argparse.Namespace) -> None:
    """Decode test scores with the best holdout configuration and write both TSVs."""
    from .infer_v3 import grouped_tsv
    best = args.decode_config or json.loads((args.stage2_dir / "evaluation.json").read_text())["best"]
    column, o2o, mode, th, mass = best.split("_")
    frame = pl.read_parquet(args.stage2_dir / "test_scored.parquet")
    candidates = frame.select("si", "ti")
    if args.calibrate:
        from .calibrate import apply_per_bin, apply_per_segment, house_segments, per_bin, per_segment
        countries = {split: pl.read_parquet(args.embed_dir / f"{split}_source1_ids.parquet")
                     .select("country").with_row_index("si") for split in ("train", "test")}
        holdout = pl.read_parquet(args.stage2_dir / "valid_predictions.parquet").join(countries["train"], on="si")
        test = frame.join(countries["test"], on="si")
        if args.calibrate_segments:
            live = test.filter(pl.col(column) >= 0.001)
            holdout = house_segments(holdout, args.cache_dir, "train")
            live = house_segments(live, args.cache_dir, "test")
            fits = per_segment(holdout, live, column, 0.02, args.calibration_bins)
            live = apply_per_segment(live, column, fits, 0.02).drop("segment")
            frame = pl.concat([live, test.filter(pl.col(column) < 0.001)], how="diagonal_relaxed").drop("country")
        else:
            fits = per_bin(holdout, test, column, 0.02, args.calibration_bins)
            frame = apply_per_bin(test, column, fits, 0.02).drop("country")
        (args.output_dir / "calibration.json").write_text(json.dumps(fits, indent=1))
    if o2o == "o2o":
        frame = one_to_one(frame, column)
    chosen = decode(frame["si"].to_numpy(), frame["ti"].to_numpy(), frame[column].to_numpy(),
                    mode, float(th), float(mass))
    s_ids, t_ids = id_frames(args.embed_dir, "test")
    matches = pl.DataFrame({
        "si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
        "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64),
    }).with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
    name = lambda pairs: (pairs.join(s_ids.rename({"entity_id": "source1_entity_id"}), on="si")
                          .join(t_ids.rename({"entity_id": "candidate_entity_id"}), on="ti"))
    source1 = s_ids["entity_id"]
    out = args.output_dir
    grouped_tsv(source1, name(candidates), "candidate_entity_ids", out / "candidate_pairs.tsv")
    grouped_tsv(source1, name(matches), "matched_entity_ids", out / "matching_results.tsv")
    print(f"decoded with {best}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="+", choices=("stage1", "p1_train", "p1_test", "stage2",
                                                     "p2_train", "p2_test", "evaluate", "write"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--model-dir", type=Path, default=Path("output/v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/v4"))
    parser.add_argument("--pair-house", action="store_true",
                        help="Add address-wide pairwise house features to stage 1")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply the per-bin label-shift correction before decoding test")
    parser.add_argument("--calibrate-segments", action="store_true",
                        help="Calibrate separately for equal, conflicting and unknown house numbers")
    parser.add_argument("--calibration-bins", type=int, default=25)
    parser.add_argument("--decode-config",
                        help="Override the holdout-selected decode, e.g. p1_o2o_expected_0.05_0.0")
    parser.add_argument("--stage2-dir", type=Path,
                        help="Where stage-2 artifacts go (defaults to --model-dir)")
    parser.add_argument("--extra", action="append", default=[],
                        help="Extra stage-2 feature parquet keyed by si, ti; '{split}' is substituted")
    parser.add_argument("--train-buckets", type=int, default=100)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--eta", type=float, default=0.05)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.stage2_dir = args.stage2_dir or args.model_dir
    args.stage2_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        started = time.time()
        {"stage1": lambda: train_stage1(args),
         "p1_train": lambda: score_stage1(args, "train"),
         "p1_test": lambda: score_stage1(args, "test"),
         "stage2": lambda: train_stage2(args),
         "p2_train": lambda: score_stage2(args, "train"),
         "p2_test": lambda: score_stage2(args, "test"),
         "evaluate": lambda: evaluate(args),
         "write": lambda: write_test(args)}[step]()
        print(f"step {step} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
