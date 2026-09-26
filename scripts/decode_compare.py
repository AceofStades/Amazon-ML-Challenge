"""Compare approximate and exact expected F0.5 with full truth and global p1 ownership."""

import argparse
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, "src")
from amzn.train_v3 import decode  # noqa: E402


def metric(chosen: dict, frame: pl.DataFrame, truth: pl.DataFrame) -> dict:
    pairs = [(source, target) for source, targets in chosen.items() for target in targets]
    picked = pl.DataFrame(pairs, schema={"si": pl.UInt32, "ti": pl.UInt32}, orient="row")
    counts = picked.join(frame.select("si", "ti", "label"), on=["si", "ti"]).group_by("si").agg(
        pl.len().alias("predicted"), pl.col("label").sum().alias("tp"))
    values = truth.join(counts, on="si", how="left").fill_null(0)
    actual, predicted, tp = (values[column].to_numpy() for column in ("actual", "predicted", "tp"))
    scores = np.divide(1.25 * tp, 0.25 * actual + predicted,
                       out=np.ones(len(values)), where=(actual + predicted) > 0)
    return {"macro_f05": float(scores.mean()), "entities": len(values), "predicted": int(predicted.sum()),
            "tp": int(tp.sum()), "fp": int((predicted - tp).sum()), "fn": int((actual - tp).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("output/cv_pre1.parquet"))
    parser.add_argument("--global-p1", default="output/v7/train_p1/*.parquet")
    parser.add_argument("--bucket-start", type=int, default=50)
    parser.add_argument("--bucket-end", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("output/decode_exact_holdout.json"))
    args = parser.parse_args()
    ids = pl.read_parquet("cache/embeddings/train_source1_ids.parquet", columns=["entity_id"]).with_row_index("si")
    connection = duckdb.connect()
    connection.register("ids", ids)
    selected = connection.execute(
        "SELECT si, entity_id FROM ids WHERE hash(entity_id) % 1000 >= ? AND hash(entity_id) % 1000 < ?",
        [args.bucket_start, args.bucket_end]).pl().with_columns(pl.col("si").cast(pl.UInt32))
    held = pl.scan_parquet(args.scores).join(selected.select("si").lazy(), on="si", how="semi").collect()
    targets = held.filter(pl.col("p1") >= 0.001).select("ti").unique()
    rivals = (pl.scan_parquet(args.global_p1).filter(pl.col("p1") >= 0.001)
              .join(targets.lazy(), on="ti", how="semi").collect(engine="streaming"))
    winners = (rivals.sort(["p1", "si"], descending=[True, False]).unique("ti", keep="first")
               .select("si", "ti").with_columns(pl.lit(True).alias("winner")))
    del rivals
    held = held.join(winners, on=["si", "ti"], how="left").with_columns(
        pl.when(pl.col("winner").fill_null(False)).then(pl.col("p1")).otherwise(0).alias("probability"))
    truth = (pl.scan_csv("dataset/train/train_ground_truth.tsv", separator="\t", quote_char=None,
                         infer_schema=False, missing_utf8_is_empty_string=True)
             .join(selected.lazy(), left_on="source1_entity_id", right_on="entity_id")
             .select("si", pl.when(pl.col("matched_entity_ids") == "").then(0)
                     .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1).alias("actual"))
             .collect())
    if truth.height != selected.height:
        raise ValueError("ground truth does not cover every evaluated entity")
    results = {}
    for mode in ("expected", "expected-exact"):
        chosen = decode(held["si"].to_numpy(), held["ti"].to_numpy(), held["probability"].to_numpy(), mode, 0.05)
        results[mode] = metric(chosen, held, truth)
        print(mode, json.dumps(results[mode]), flush=True)
    results["delta"] = results["expected-exact"]["macro_f05"] - results["expected"]["macro_f05"]
    results["scope"] = "stage1 p1; global training target winners; full truth; held-out source hash buckets"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"macro F0.5 change {results['delta']:+.8f}", flush=True)


if __name__ == "__main__":
    main()
