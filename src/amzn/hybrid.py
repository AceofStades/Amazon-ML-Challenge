"""Choose the scorer per country: models learned from labelled countries do not
transfer everywhere.

The cross-encoder blend and stage-2 context features were fitted on US and India,
where business names are distinctive. In a country without training labels
(France in the test set) names are generic ("Lille Ecole SARL" exists many times),
and those models over-trust name agreement. For every country present in the
labelled holdout the blend score is used; for any other country the stage-1 score
is used. Each group is calibrated against the holdout distribution of its own
scorer (house-relation segments), then one target per entity and expected-F0.5
decoding are applied as usual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .calibrate import apply_per_segment, house_segments, per_segment
from .infer_v3 import grouped_tsv
from .train_v3 import decode
from .train_v4 import one_to_one
from .v3 import id_frames


def calibrated(holdout: pl.DataFrame, test: pl.DataFrame, column: str, cache_dir: Path,
               mode: str = "segment") -> pl.DataFrame:
    live = test.filter(pl.col(column) >= 0.001)
    if mode == "none":
        return live
    if mode == "uniform":
        from .calibrate import apply_per_bin, per_bin
        return apply_per_bin(live, column, per_bin(holdout, live, column, 0.02, 25), 0.02)
    live = house_segments(live, cache_dir, "test")
    held = house_segments(holdout, cache_dir, "train")
    if mode == "segment-name":
        from .calibrate import name_segments
        live = name_segments(live, cache_dir, "test")
        held = name_segments(held, cache_dir, "train")
    fits = per_segment(held, live, column, 0.02, 25)
    return apply_per_segment(live, column, fits, 0.02)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend-dir", type=Path, default=Path("output/v4_blend"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/hybrid"))
    parser.add_argument("--calibration", choices=("segment", "segment-name"), default="segment",
                        help="Label-shift correction for countries with labels")
    parser.add_argument("--fallback-calibration", choices=("segment", "segment-name", "uniform", "none"),
                        default="segment",
                        help="Label-shift correction for countries without labels")
    parser.add_argument("--fallback-dir", type=Path,
                        help="Run whose stage-1 scores serve countries without labels (default: --blend-dir)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    countries = {split: pl.read_parquet(args.embed_dir / f"{split}_source1_ids.parquet")
                 .select("country").with_row_index("si") for split in ("train", "test")}
    holdout = pl.read_parquet(args.blend_dir / "valid_predictions.parquet").join(countries["train"], on="si")
    test = pl.read_parquet(args.blend_dir / "test_scored.parquet").join(countries["test"], on="si")
    fallback_dir = args.fallback_dir or args.blend_dir
    fallback_holdout = pl.read_parquet(fallback_dir / "valid_predictions.parquet").join(countries["train"], on="si")
    fallback_test = pl.read_parquet(fallback_dir / "test_scored.parquet").join(countries["test"], on="si")
    labelled = set(holdout["country"].unique().to_list())
    print("countries with labels:", sorted(labelled), flush=True)

    known = test.filter(pl.col("country").is_in(list(labelled)))
    unknown = fallback_test.filter(~pl.col("country").is_in(list(labelled)))
    parts = []
    if known.height:
        raw = known.select("si", "ti", pl.col("p2").alias("raw"))
        parts.append(calibrated(holdout, known, "p2", args.cache_dir, args.calibration)
                     .select("si", "ti", pl.col("p2").alias("p"))
                     .join(raw, on=["si", "ti"]))
    if unknown.height:
        raw = unknown.select("si", "ti", pl.col("p1").alias("raw"))
        parts.append(calibrated(fallback_holdout, unknown, "p1", args.cache_dir, args.fallback_calibration)
                     .select("si", "ti", pl.col("p1").alias("p")).join(raw, on=["si", "ti"]))
    scored = one_to_one(pl.concat(parts), "p", tie_break="raw")
    chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p"].to_numpy(), "expected", 0.05, 0.0)

    s_ids, t_ids = id_frames(args.embed_dir, "test")
    matches = pl.DataFrame({
        "si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
        "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64),
    }).with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
    name = lambda pairs: (pairs.join(s_ids.rename({"entity_id": "source1_entity_id"}), on="si")
                          .join(t_ids.rename({"entity_id": "candidate_entity_id"}), on="ti"))
    grouped_tsv(s_ids["entity_id"], name(test.select("si", "ti")), "candidate_entity_ids",
                args.output_dir / "candidate_pairs.tsv")
    grouped_tsv(s_ids["entity_id"], name(matches), "matched_entity_ids", args.output_dir / "matching_results.tsv")
    scored.write_parquet(args.output_dir / "test_calibrated.parquet")


if __name__ == "__main__":
    main()
