#!/usr/bin/env python3
"""Score bounded chunks in fresh GPU processes and resume validated checkpoints."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def fingerprint(path: Path, *, hash_contents: bool = True) -> dict:
    stat = path.stat()
    identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        identity["sha256"] = digest.hexdigest()
    return identity


def checkpoint_identity(args: argparse.Namespace, pairs: pq.ParquetFile) -> dict:
    model_files = sorted(path for path in args.model_dir.rglob("*") if path.is_file())
    if not model_files:
        raise ValueError(f"no model files in {args.model_dir}")
    return {
        "version": 1,
        "split": args.split,
        "pairs": fingerprint(args.pairs),
        "rows": pairs.metadata.num_rows,
        "pair_schema": str(pairs.schema_arrow),
        "model": [fingerprint(path) for path in model_files],
        # Dataset stat changes invalidate the index-to-record mapping.
        "data": [fingerprint(args.data_dir / f"{args.split}_source{i}.tsv", hash_contents=False)
                 for i in (1, 2, 3)],
        "batch_size": args.batch_size,
        "chunk_rows": args.chunk_rows,
    }


def valid_scores(path: Path, expected: pa.Table) -> bool:
    try:
        scores = pq.read_table(path, columns=["si", "ti", "ce_logit"])
        return (
            scores.num_rows == expected.num_rows
            and all(scores[column].equals(expected[column]) for column in ("si", "ti"))
            and pa.types.is_floating(scores["ce_logit"].type)
            and scores["ce_logit"].null_count == 0
            and pc.all(pc.is_finite(scores["ce_logit"])).as_py() is True
        )
    except (OSError, pa.ArrowException):
        return False


def recover(args: argparse.Namespace) -> None:
    for name in ("pairs", "output", "model_dir", "data_dir", "checkpoints"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    with (args.checkpoints / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns these checkpoints") from error
        pairs = pq.ParquetFile(args.pairs)
        if pairs.metadata.num_rows == 0 or not {"si", "ti"}.issubset(pairs.schema_arrow.names):
            raise ValueError("pairs must contain nonempty si and ti columns")
        identity = checkpoint_identity(args, pairs)
        manifest = args.checkpoints / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("incompatible checkpoints; use a new --checkpoints directory")
        else:
            temporary = manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(identity, indent=2) + "\n")
            temporary.replace(manifest)

        environment = os.environ.copy()
        source = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        environment["TOKENIZERS_PARALLELISM"] = "false"
        chunks = []
        completed = 0
        for number, batch in enumerate(pairs.iter_batches(batch_size=args.chunk_rows, columns=["si", "ti"])):
            expected = pa.Table.from_batches([batch])
            chunk = args.checkpoints / f"part-{number:06d}.parquet"
            chunks.append(chunk)
            if valid_scores(chunk, expected):
                print(f"reuse chunk {number}: {expected.num_rows:,} pairs", flush=True)
            else:
                inputs = chunk.with_suffix(".pairs.parquet")
                pending = chunk.with_suffix(".partial.parquet")
                pq.write_table(expected, inputs, compression="zstd")
                print(f"score chunk {number}: {expected.num_rows:,} pairs, batch {args.batch_size}", flush=True)
                subprocess.run([
                    sys.executable, "-m", "amzn.crossenc", "score",
                    "--split", args.split, "--data-dir", str(args.data_dir),
                    "--model-dir", str(args.model_dir), "--pairs", str(inputs),
                    "--output", str(pending), "--batch-size", str(args.batch_size),
                ], check=True, env=environment)
                if not valid_scores(pending, expected):
                    raise RuntimeError(f"invalid scores from chunk {number}")
                pending.replace(chunk)
                inputs.unlink()
            completed += expected.num_rows
            print(f"checkpointed {completed:,}/{pairs.metadata.num_rows:,} pairs", flush=True)

        pending_output = args.output.with_name(args.output.name + ".recovering")
        with pq.ParquetWriter(pending_output, pq.read_schema(chunks[0]), compression="zstd") as writer:
            for chunk in chunks:
                writer.write_table(pq.read_table(chunk))
        if pq.read_metadata(pending_output).num_rows != pairs.metadata.num_rows:
            raise RuntimeError("assembled score row count differs from input")
        pending_output.replace(args.output)
        print(f"wrote {args.output}: {completed:,} pairs", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "test"))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.chunk_rows < 1:
        parser.error("batch size and chunk rows must be positive")
    args.data_dir = args.data_dir or Path("dataset") / args.split
    args.checkpoints = args.checkpoints or args.output.parent / f".{args.output.stem}.chunks"
    try:
        recover(args)
    except subprocess.CalledProcessError as error:
        print(f"scoring failed with exit {error.returncode}; completed chunks remain resumable", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, pa.ArrowException) as error:
        print(f"recovery failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
