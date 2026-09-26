#!/usr/bin/env python3
"""Check that every expected model has readable train and test score footers."""

import argparse
import csv
from pathlib import Path
import re
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def model_names(fleet: Path, local: list[str]) -> list[str]:
    names = []
    with fleet.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4 or any(not cell.strip() for cell in row[:4]):
                raise ValueError(f"line {reader.line_num}: expected four nonempty columns")
            name = row[3].strip()
            if name in names:
                raise ValueError(f"line {reader.line_num}: duplicate cache name {name}")
            names.append(name)
    if not names:
        raise ValueError("no models listed")
    for name in local:
        if name not in names:
            names.append(name)
    for name in names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"invalid cache name {name!r}")
    return names


def scores_ready(cache: Path) -> bool:
    for split in ("train", "test"):
        try:
            pq.read_metadata(cache / f"{split}_scores.parquet")
        except (OSError, pa.ArrowException):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", type=Path, default=Path("kaggle/fleet.tsv"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--local", action="append", default=[], metavar="NAME")
    args = parser.parse_args(argv)
    try:
        names = model_names(args.fleet, args.local)
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        print(f"invalid fleet: {error}", file=sys.stderr)
        return 2
    pending = [name for name in names if not scores_ready(args.cache_root / f"ce_{name}")]
    if pending:
        print(f"pending: {', '.join(pending)}")
        return 1
    print(f"ready: {len(names)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
