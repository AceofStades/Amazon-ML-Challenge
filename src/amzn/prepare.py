"""Prepare raw challenge TSV files for blocking and model features."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .normalize import RECORD_COLUMNS, normalize_record


def prepare_file(input_path: Path, output_path: Path, limit: int | None = None) -> tuple[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    count = 0
    with input_path.open(newline="", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.writer(destination, delimiter="\t")
        writer.writerow(RECORD_COLUMNS)
        for row in reader:
            writer.writerow(normalize_record(row))
            count += 1
            if limit is not None and count >= limit:
                break
    temporary.replace(output_path)
    return str(output_path), count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    splits = ("train", "test") if args.split == "both" else (args.split,)
    tasks = [
        (args.data_dir / split / f"{split}_source{source}.tsv",
         args.cache_dir / f"{split}_source{source}.tsv", args.limit)
        for split in splits for source in (1, 2, 3)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare_file, *task) for task in tasks]
        for future in as_completed(futures):
            path, count = future.result()
            print(f"Prepared {count:,} records: {path}", flush=True)


if __name__ == "__main__":
    main()
