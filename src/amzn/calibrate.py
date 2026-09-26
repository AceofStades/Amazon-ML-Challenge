"""Label-shift correction of pair probabilities for the test set (no test labels).

Test contains about twice as many look-alike decoy records per Source 1 entity as
the training data, while the confident matches per entity are unchanged. With
the class-conditional score densities f1, f0 taken from the labelled holdout,
the test score histogram m(b) per entity is fitted per country as
    m(b) = a * n1(b) + c * n0(b)
over bins of the logit score (b above --min-p). The corrected posterior of a
test pair in bin b is a*n1(b) / (a*n1(b) + c*n0(b)), i.e. the holdout
posterior with its odds multiplied by a/c. Countries absent from the holdout
use the pooled holdout densities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import nnls


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def histograms(scores: np.ndarray, labels: np.ndarray | None, edges: np.ndarray, entities: int):
    index = np.digitize(logit(scores), edges)
    bins = len(edges) + 1
    if labels is None:
        return np.bincount(index, minlength=bins) / entities
    pos = np.bincount(index, weights=labels, minlength=bins) / entities
    neg = np.bincount(index, weights=1 - labels, minlength=bins) / entities
    return pos, neg


def fit(holdout: pl.DataFrame, test: pl.DataFrame, column: str, min_p: float, bins: int) -> dict:
    edges = np.linspace(logit(np.array([min_p]))[0], logit(np.array([0.999]))[0], bins)
    result = {}
    pooled = histograms(holdout[column].to_numpy(), holdout["label"].to_numpy().astype(float), edges,
                        holdout["si"].n_unique())
    for country in sorted(test["country"].unique().to_list()):
        h = holdout.filter(pl.col("country") == country)
        n1, n0 = (histograms(h[column].to_numpy(), h["label"].to_numpy().astype(float), edges, h["si"].n_unique())
                  if h.height else pooled)
        t = test.filter(pl.col("country") == country)
        m = histograms(t[column].to_numpy(), None, edges, t["si"].n_unique())
        use = slice(1, None)  # bin 0 holds scores below min_p
        (a, c), residual = nnls(np.stack([n1[use], n0[use]], 1), m[use])
        result[country] = {
            "a": float(a), "c": float(c), "odds_multiplier": float(a / max(c, 1e-9)),
            "residual": float(residual), "holdout_country": bool(h.height),
            "bins": {"test": m.round(4).tolist(), "pos": n1.round(4).tolist(), "neg": n0.round(4).tolist()},
        }
        print(f"{country}: positives x{a:.3f}, negatives x{c:.3f}, odds x{a / max(c, 1e-9):.3f}, residual {residual:.4f}",
              flush=True)
    return result


def apply(frame: pl.DataFrame, column: str, fits: dict, min_p: float) -> pl.DataFrame:
    multiplier = pl.col("country").replace_strict(
        {k: v["odds_multiplier"] for k, v in fits.items()}, default=1.0, return_dtype=pl.Float64)
    odds = pl.col(column) / (1 - pl.col(column)).clip(1e-9)
    adjusted = (odds * multiplier) / (1 + odds * multiplier)
    return frame.with_columns(
        pl.when(pl.col(column) >= min_p).then(adjusted).otherwise(pl.col(column)).cast(pl.Float32).alias(column))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True, help="parquet with si, label, score")
    parser.add_argument("--test", type=Path, required=True, help="parquet with si, ti, score")
    parser.add_argument("--column", default="p2")
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--min-p", type=float, default=0.02)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    countries = {split: pl.read_parquet(args.embed_dir / f"{split}_source1_ids.parquet")
                 .select("country").with_row_index("si") for split in ("train", "test")}
    holdout = pl.read_parquet(args.holdout).join(countries["train"], on="si")
    test = pl.read_parquet(args.test).join(countries["test"], on="si")
    fits = fit(holdout, test, args.column, args.min_p, args.bins)
    apply(test, args.column, fits, args.min_p).drop("country").write_parquet(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(fits, indent=1))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
