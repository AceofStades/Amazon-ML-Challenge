"""Full-competition target context for every cross-encoder-scored pair (the blend's pairs), as a blend extra:
cache/ctx/{split}.parquet with q1 (stage-1 score over the sum for all candidate entities of the target), m1, the best
competitor's score, and raw token-set agreement of the names (and how many candidates share it).
Usage: python scripts/ctx_features.py train|test"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "scripts")
from compose import extra_features  # noqa: E402

split = sys.argv[1]
pairs = pl.read_parquet(f"cache/ce/{split}_scores.parquet", columns=["si", "ti"]).with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
out = extra_features(pairs, split).select("si", "ti", "q1", "m1", "top_other1", "s_tok_eq", "n_s_tok_eq")
Path("cache/ctx").mkdir(parents=True, exist_ok=True)
out.write_parquet(f"cache/ctx/{split}.parquet")
print(split, out.height, out.select(pl.col("q1").mean(), pl.col("m1").mean(), pl.col("s_tok_eq").mean()).rows())
