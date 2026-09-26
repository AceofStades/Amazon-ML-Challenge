"""Export record texts and pair lists for cross-encoder jobs on Kaggle GPUs.

Texts follow the embedding index order: Source 1 rows by si, Source 2 then
Source 3 rows by ti. Pair files hold integer (si, ti) positions only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl

from .embed import clean, read_source


def texts(frame: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        "text": [f"{clean(n)} | {clean(a)}" for n, a in zip(frame["business_name"], frame["business_address"])],
        "country": frame["country"],
    })


def export_texts(data_root: Path, out: Path) -> None:
    for split in ("train", "test"):
        folder = data_root / split
        texts(read_source(folder / f"{split}_source1.tsv")).write_parquet(
            out / f"{split}_s1.parquet", compression="zstd", compression_level=9)
        targets = pl.concat([read_source(folder / f"{split}_source{i}.tsv") for i in (2, 3)])
        texts(targets).write_parquet(out / f"{split}_t.parquet", compression="zstd", compression_level=9)
        print(f"exported {split} texts", flush=True)


def export_training_pairs(embed_dir: Path, truth_path: Path, out: Path, entities: int, depth: int,
                          min_bucket: int) -> None:
    s_ids = pl.read_parquet(embed_dir / "train_source1_ids.parquet").select("entity_id").with_row_index("si")
    t_ids = pl.concat([pl.read_parquet(embed_dir / f"train_source{i}_ids.parquet").select("entity_id")
                       for i in (2, 3)]).with_row_index("ti")
    buckets = duckdb.sql("SELECT si, hash(entity_id) % 1000 AS bucket FROM s_ids").pl()
    chosen = buckets.filter(pl.col("bucket") >= min_bucket).sample(n=entities, seed=11)["si"]
    knn = (pl.scan_parquet(embed_dir / "train_knn.parquet")
           .filter((pl.col("rank") <= depth) & pl.col("q").is_in(chosen.implode()))
           .select(pl.col("q").cast(pl.Int64).alias("si"), pl.col("d").cast(pl.Int64).alias("ti")).collect())
    truth = (read_source(truth_path).with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .join(s_ids, left_on="source1_entity_id", right_on="entity_id")
             .join(t_ids, left_on="matched_entity_ids", right_on="entity_id")
             .select(pl.col("si").cast(pl.Int64), pl.col("ti").cast(pl.Int64), pl.lit(1, pl.Int8).alias("label")))
    positives = truth.filter(pl.col("si").is_in(chosen.implode())).select("si", "ti")
    pairs = (pl.concat([knn, positives]).unique()
             .join(truth, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
             .sample(fraction=1.0, shuffle=True, seed=12))
    pairs.write_parquet(out / "ce_train_pairs.parquet", compression="zstd")
    print(f"training pairs {pairs.height:,}, positive rate {pairs['label'].mean():.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--truth", type=Path, default=Path("dataset/train/train_ground_truth.tsv"))
    parser.add_argument("--output", type=Path, default=Path("kaggle/data"))
    parser.add_argument("--entities", type=int, default=600_000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--min-bucket", type=int, default=100)
    parser.add_argument("--skip-texts", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_texts:
        export_texts(args.data_root, args.output)
    export_training_pairs(args.embed_dir, args.truth, args.output, args.entities, args.depth, args.min_bucket)


if __name__ == "__main__":
    main()
