"""Two-stage GPU XGBoost matcher over v3 candidates with set-level decoding.

Stage 1 scores each pair from string and embedding features, out-of-fold on
the training entities. Stage 2 adds context from stage-1 scores of the other
candidates of the same Source 1 entity (rank, margin, sibling support).
Decoding picks, per Source 1 entity, the prefix of candidates ranked by
probability that maximises expected F0.5 (the empty set included).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import torch
import xgboost as xgb


ID_COLUMNS = ("source1_entity_id", "candidate_entity_id", "si", "ti")
STAGE2_EXTRA = (
    "p1", "p1_rank", "p1_max", "p1_second", "p1_sum", "p1_count50", "p1_minus_max",
    "p1_sib_max", "p1_sib_sum",
)


def load_split(cache_dir: Path, split: str) -> pl.DataFrame:
    frame = pl.read_parquet(cache_dir / "v3" / split / "*.parquet")
    buckets = duckdb.sql(
        "SELECT hash(source1_entity_id) % 1000 AS bucket FROM frame"
    ).pl()["bucket"]
    return frame.with_columns(bucket=buckets).sort("si", "ti")


def attach_labels(frame: pl.DataFrame, truth_path: Path) -> pl.DataFrame:
    truth = (
        pl.read_csv(truth_path, separator="\t", quote_char=None, infer_schema=False,
                    missing_utf8_is_empty_string=True)
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .select(pl.col("source1_entity_id"), pl.col("matched_entity_ids").alias("candidate_entity_id"),
                pl.lit(1, pl.Int8).alias("label"))
    )
    return frame.join(truth, on=["source1_entity_id", "candidate_entity_id"], how="left").with_columns(
        pl.col("label").fill_null(0))


def truth_sets(truth_path: Path, ids: set[str]) -> dict[str, set[str]]:
    table = pl.read_csv(truth_path, separator="\t", quote_char=None, infer_schema=False,
                        missing_utf8_is_empty_string=True)
    table = table.filter(pl.col("source1_entity_id").is_in(list(ids)))
    return {s: set(m.split(",")) if m else set()
            for s, m in zip(table["source1_entity_id"], table["matched_entity_ids"])}


def feature_names(frame: pl.DataFrame, extra: tuple[str, ...] = ()) -> list[str]:
    skip = set(ID_COLUMNS) | {"bucket", "label", *extra}
    return [c for c in frame.columns if c not in skip] + list(extra)


def params(args: argparse.Namespace) -> dict:
    return {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "device": "cuda", "max_depth": args.max_depth, "eta": args.eta, "subsample": 0.85,
        "colsample_bytree": 0.8, "min_child_weight": 3, "max_bin": 256, "seed": 42,
    }


def matrix(frame: pl.DataFrame, names: list[str], label: bool = True) -> xgb.DMatrix:
    X = frame.select(names).cast(pl.Float32).to_numpy()
    return xgb.DMatrix(X, label=frame["label"].to_numpy() if label else None, feature_names=names)


def fit(train: pl.DataFrame, valid: pl.DataFrame | None, names: list[str], args, rounds: int):
    dtrain = matrix(train, names)
    evals = [(matrix(valid, names), "valid")] if valid is not None else []
    return xgb.train(params(args), dtrain, num_boost_round=rounds, evals=evals,
                     early_stopping_rounds=50 if evals else None, verbose_eval=100)


def predict(model: xgb.Booster, frame: pl.DataFrame, names: list[str]) -> np.ndarray:
    best = getattr(model, "best_iteration", None)
    kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
    return model.predict(matrix(frame, names, label=False), **kwargs)


def group_bounds(si: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    return starts, np.r_[starts[1:], len(si)]


@torch.inference_mode()
def stage2_context(frame: pl.DataFrame, p: np.ndarray, t_vec: np.ndarray,
                   max_group: int = 160, batch: int = 2048) -> pl.DataFrame:
    frame = frame.with_columns(p1=p.astype(np.float32))
    grouped = frame.lazy().with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("si").cast(pl.Float32).alias("p1_rank"),
        pl.col("p1").max().over("si").alias("p1_max"),
        pl.col("p1").top_k(2).min().over("si").alias("p1_second"),
        pl.col("p1").sum().over("si").alias("p1_sum"),
        (pl.col("p1") > 0.5).sum().over("si").cast(pl.Float32).alias("p1_count50"),
    ).with_columns((pl.col("p1") - pl.col("p1_max")).alias("p1_minus_max")).collect()
    si = grouped["si"].to_numpy()
    ti = grouped["ti"].to_numpy()
    starts, ends = group_bounds(si)
    sib_max = np.zeros(len(si), np.float32)
    sib_sum = np.zeros(len(si), np.float32)
    for b in range(0, len(starts), batch):
        gs, ge = starts[b:b + batch], ends[b:b + batch]
        lengths = np.minimum(ge - gs, max_group)
        g = int(lengths.max())
        valid = np.arange(g)[None, :] < lengths[:, None]
        rows = np.where(valid, gs[:, None] + np.arange(g)[None, :], 0)
        vec = torch.from_numpy(np.ascontiguousarray(t_vec[ti[rows]])).cuda().float()
        prob = torch.from_numpy(np.where(valid, p[rows], 0).astype(np.float32)).cuda()
        mask = torch.from_numpy(valid).cuda()
        tt = vec @ vec.transpose(1, 2)
        keep = mask[:, :, None] & mask[:, None, :] & ~torch.eye(g, dtype=torch.bool, device="cuda")[None]
        weighted = torch.where(keep, tt.clamp(min=0) * prob[:, None, :], torch.zeros_like(tt))
        flat = rows[valid]
        sib_max[flat] = weighted.max(2).values[mask].cpu().numpy()
        sib_sum[flat] = torch.where(keep & (tt > 0.8), prob[:, None, :], torch.zeros_like(tt)).sum(2)[mask].cpu().numpy()
    return grouped.with_columns(p1_sib_max=sib_max, p1_sib_sum=sib_sum)


def decode(si: np.ndarray, targets: np.ndarray, p: np.ndarray, mode: str, threshold: float,
           recall_mass: float = 0.0) -> dict[int, list[str]]:
    """Return predicted target ids per Source 1 row index."""
    order = np.lexsort((-p, si))
    si, targets, p = si[order], targets[order], p[order]
    starts, ends = group_bounds(si)
    result: dict[int, list[str]] = {}
    for a, b in zip(starts, ends):
        probs = p[a:b]
        if mode == "threshold":
            k = int((probs >= threshold).sum())
        else:
            # Expected F0.5 ~ 1.25*TP / (0.25*A + k) with A the expected true count.
            actual = probs.sum() + recall_mass
            empty = float(np.prod(1 - probs))
            tp = np.cumsum(probs)
            ks = np.arange(1, len(probs) + 1)
            gain = 1.25 * tp / (0.25 * actual + ks) * (1 - empty)
            gain = np.where(probs >= threshold, gain, -1)
            best = int(np.argmax(gain))
            k = best + 1 if gain[best] > empty else 0
        result[int(si[a])] = targets[a:a + k].tolist()
    return result


def macro_f05(pred: dict[int, list[str]], truth: dict[str, set[str]], id_of: dict[int, str],
              all_ids: set[str]) -> float:
    by_id = {id_of[k]: set(v) for k, v in pred.items()}
    total = 0.0
    for s in all_ids:
        actual, got = truth.get(s, set()), by_id.get(s, set())
        if not actual:
            total += float(not got)
        elif got:
            c = len(actual & got)
            if c:
                P, R = c / len(got), c / len(actual)
                total += 1.25 * P * R / (0.25 * P + R)
    return total / len(all_ids)


def tune(frame: pl.DataFrame, p: np.ndarray, truth, all_ids) -> dict:
    si = frame["si"].to_numpy()
    targets = frame["candidate_entity_id"].to_numpy()
    id_of = dict(zip(frame["si"].to_list(), frame["source1_entity_id"].to_list()))
    results = {}
    for th in np.arange(0.3, 0.86, 0.05):
        results[f"threshold_{th:.2f}"] = macro_f05(decode(si, targets, p, "threshold", th), truth, id_of, all_ids)
    for th in (0.05, 0.1, 0.2, 0.3):
        for mass in (0.0, 0.2, 0.5):
            results[f"expected_{th:.2f}_{mass}"] = macro_f05(
                decode(si, targets, p, "expected", th, mass), truth, id_of, all_ids)
    best = max(results, key=results.get)
    return {"best": best, "score": results[best], "all": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/v3"))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--eta", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = attach_labels(load_split(args.cache_dir, "train"), args.truth)
    names1 = feature_names(frame)
    is_valid = (frame["bucket"] >= 50).to_numpy()
    train, valid = frame.filter(~pl.Series(is_valid)), frame.filter(pl.Series(is_valid))
    print(f"train rows {train.height:,} valid rows {valid.height:,} features {len(names1)}", flush=True)

    # Stage 1: out-of-fold scores for training entities, fold-average for validation.
    fold = (train["bucket"] % args.folds).to_numpy()
    oof = np.zeros(train.height, np.float32)
    valid_p1 = np.zeros(valid.height, np.float32)
    for f in range(args.folds):
        inner = train.filter(pl.Series(fold != f))
        model = fit(inner, valid, names1, args, args.rounds)
        oof[fold == f] = predict(model, train.filter(pl.Series(fold == f)), names1)
        valid_p1 += predict(model, valid, names1) / args.folds
        model.save_model(args.output_dir / f"stage1_fold{f}.json")
        print(f"stage1 fold {f} best iteration {model.best_iteration}", flush=True)

    t_vec = np.concatenate([np.load(args.embed_dir / f"train_source{i}.npy", mmap_mode="r") for i in (2, 3)])
    train2 = stage2_context(train, oof, t_vec)
    valid2 = stage2_context(valid, valid_p1, t_vec)
    names2 = feature_names(frame) + list(STAGE2_EXTRA)
    model2 = fit(train2, valid2, names2, args, args.rounds)
    model2.save_model(args.output_dir / "stage2.json")
    p2 = predict(model2, valid2, names2)

    valid_ids = set(duckdb.sql(
        "SELECT entity_id FROM read_csv('dataset/train/train_source1.tsv', delim='\t', quote='', header=true, all_varchar=true) "
        "WHERE hash(entity_id) % 1000 BETWEEN 50 AND 59").pl()["entity_id"].to_list())
    truth = truth_sets(args.truth, valid_ids)
    report = {
        "features_stage1": names1, "features_stage2": names2,
        "stage2_best_iteration": model2.best_iteration,
        "valid_entities": len(valid_ids),
        "candidate_recall": float(valid2["label"].sum()) / sum(len(v) for v in truth.values()),
        "stage1": tune(valid2, valid_p1, truth, valid_ids),
        "stage2": tune(valid2, p2, truth, valid_ids),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("features")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
