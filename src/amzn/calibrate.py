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


def house_segments(frame: pl.DataFrame, cache_dir: Path, split: str) -> pl.DataFrame:
    """Attach the house-number relation of each pair: equal (the Source 1 house number
    appears among the target's address numbers), conflict (both have numbers, no match)
    or unknown. Test decoys sit at nearby house numbers, so they concentrate in 'conflict'."""
    from .train_v4 import house_numbers
    s_house, _, t_numbers = house_numbers(cache_dir, split)
    hs = pl.Series("hs", s_house[frame["si"].to_numpy()])
    nums = t_numbers.gather(frame["ti"].to_numpy())
    relation = pl.DataFrame({"hs": hs, "nums": nums}).select(
        pl.when((pl.col("hs") != "") & pl.col("nums").list.contains(pl.col("hs"))).then(pl.lit("equal"))
        .when((pl.col("hs") != "") & (pl.col("nums").list.len() > 0)).then(pl.lit("conflict"))
        .otherwise(pl.lit("unknown")).alias("segment"))
    return frame.with_columns(relation["segment"])


def missing_segments(frame: pl.DataFrame, cache_dir: Path, split: str) -> pl.DataFrame:
    """Split 'unknown' into targets without any address and the rest: the test holds more same-name decoys
    without an address than the training data, while records without a house number are not affected."""
    empty = pl.concat([pl.read_csv(cache_dir / f"{split}_source{i}.tsv", separator="\t", quote_char=None,
                                   infer_schema=False, columns=["address"]) for i in (2, 3)])["address"].fill_null("") == ""
    flag = pl.Series("t_empty", empty.to_numpy()[frame["ti"].to_numpy()])
    return frame.with_columns(
        pl.when((pl.col("segment") == "unknown") & flag).then(pl.lit("unknown_noaddr"))
        .otherwise(pl.col("segment")).alias("segment"))


def name_segments(frame: pl.DataFrame, cache_dir: Path, split: str, threshold: float = 0.95) -> pl.DataFrame:
    """Refine 'segment' by name agreement (IDF cosine of canonical names >= threshold):
    decoys usually carry a perturbed name, true records usually the same one."""
    import glob
    keys = frame.select("si", "ti")
    parts = [pl.read_parquet(path, columns=["si", "ti", "name_idf_cos"]).join(keys, on=["si", "ti"], how="semi")
             for path in sorted(glob.glob(str(cache_dir / "v3" / split / "part*.parquet")))]
    names = pl.concat(parts)
    frame = frame.join(names, on=["si", "ti"], how="left")
    return frame.with_columns(
        (pl.col("segment") + pl.when(pl.col("name_idf_cos").fill_null(0) >= threshold)
         .then(pl.lit("_samename")).otherwise(pl.lit("_othername"))).alias("segment")).drop("name_idf_cos")


SEGMENTS = [f"{house}_{name}" for house in ("equal", "conflict", "unknown") for name in ("samename", "othername")]


def per_segment(holdout: pl.DataFrame, test: pl.DataFrame, column: str, min_p: float, bins: int,
                top_p: float = 0.99) -> dict:
    """Per-bin correction within each (country, segment); positives per entity keep their
    holdout density, scaled by the country factor a fitted on the top bins of all segments."""
    edges = np.linspace(logit(np.array([min_p]))[0], logit(np.array([0.999]))[0], bins)
    top = np.flatnonzero(np.r_[-np.inf, edges] >= logit(np.array([top_p]))[0])
    result = {}
    for country in sorted(test["country"].unique().to_list()):
        h = holdout.filter(pl.col("country") == country)
        if not h.height:
            h = holdout
        t = test.filter(pl.col("country") == country)
        nh, nt = h["si"].n_unique(), t["si"].n_unique()
        n1_all, _ = histograms(h[column].to_numpy(), h["label"].to_numpy().astype(float), edges, nh)
        m_all = histograms(t[column].to_numpy(), None, edges, nt)
        a = float(m_all[top].sum() / max(n1_all[top].sum(), 1e-9))
        present = sorted(set(h["segment"].unique().to_list()) | set(t["segment"].unique().to_list()))
        for segment in present:
            hs = h.filter(pl.col("segment") == segment)
            ts = t.filter(pl.col("segment") == segment)
            n1, _ = histograms(hs[column].to_numpy(), hs["label"].to_numpy().astype(float), edges, nh)
            m = histograms(ts[column].to_numpy(), None, edges, nt)
            posterior = np.clip(a * n1 / np.maximum(m, 1e-9), 0, 1)
            posterior[m < 1e-6] = np.nan
            # Empty cells borrow from the neighbour below; then force monotone.
            posterior = pl.Series(posterior).fill_nan(None).fill_null(strategy="forward").fill_null(0.0).to_numpy().copy()
            posterior[0] = min(posterior[0], posterior[1])
            posterior = np.maximum.accumulate(posterior)
            result[f"{country}|{segment}"] = {"a": a, "edges": edges.tolist(), "posterior": posterior.round(4).tolist()}
            print(country, segment, f"a={a:.3f}", np.round(posterior[1:], 2).tolist(), flush=True)
    return result


def apply_per_segment(frame: pl.DataFrame, column: str, fits: dict, min_p: float) -> pl.DataFrame:
    frame = frame.with_columns((pl.col("country") + "|" + pl.col("segment")).alias("cell"))
    renamed = frame.drop("country").rename({"cell": "country"})
    adjusted = apply_per_bin(renamed, column, fits, min_p)
    return adjusted.rename({"country": "cell"}).drop("cell")


def per_bin(holdout: pl.DataFrame, test: pl.DataFrame, column: str, min_p: float, bins: int,
            top_p: float = 0.99) -> dict:
    """Positives per entity keep their holdout density (scaled by a, fitted on the top
    bins); every extra test pair in a bin is a negative. Posterior per bin is
    a * n1(b) / m(b), made non-decreasing in the score."""
    edges = np.linspace(logit(np.array([min_p]))[0], logit(np.array([0.999]))[0], bins)
    pooled = histograms(holdout[column].to_numpy(), holdout["label"].to_numpy().astype(float), edges,
                        holdout["si"].n_unique())
    top = np.flatnonzero(np.r_[-np.inf, edges] >= logit(np.array([top_p]))[0])
    result = {}
    for country in sorted(test["country"].unique().to_list()):
        h = holdout.filter(pl.col("country") == country)
        n1, _ = (histograms(h[column].to_numpy(), h["label"].to_numpy().astype(float), edges, h["si"].n_unique())
                 if h.height else pooled)
        t = test.filter(pl.col("country") == country)
        m = histograms(t[column].to_numpy(), None, edges, t["si"].n_unique())
        a = float(m[top].sum() / max(n1[top].sum(), 1e-9))
        posterior = np.clip(a * n1 / np.maximum(m, 1e-9), 0, 1)
        posterior[0] = min(posterior[0], posterior[1])
        posterior = np.maximum.accumulate(posterior)
        result[country] = {"a": a, "edges": edges.tolist(), "posterior": posterior.round(4).tolist()}
        print(country, f"a={a:.3f}", "posterior by bin:", np.round(posterior[1:], 2).tolist(), flush=True)
    return result


def apply_per_bin(frame: pl.DataFrame, column: str, fits: dict, min_p: float) -> pl.DataFrame:
    parts = []
    for country, group in frame.group_by("country", maintain_order=True):
        fit_ = fits.get(country[0])
        if fit_ is None:
            parts.append(group)
            continue
        edges = np.asarray(fit_["edges"])
        posterior = np.asarray(fit_["posterior"])
        # Interpolate between bin centres in logit space; bin 0 is below min_p.
        centres = np.r_[edges[0] - 1.0, (edges[:-1] + edges[1:]) / 2, edges[-1] + 1.0]
        scores = group[column].to_numpy()
        adjusted = np.interp(logit(scores), centres, posterior)
        adjusted = np.where(scores >= min_p, adjusted, np.minimum(scores, adjusted))
        parts.append(group.with_columns(pl.Series(column, adjusted.astype(np.float32))))
    return pl.concat(parts)


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
    parser.add_argument("--mode", choices=("odds", "per-bin"), default="per-bin")
    args = parser.parse_args()
    countries = {split: pl.read_parquet(args.embed_dir / f"{split}_source1_ids.parquet")
                 .select("country").with_row_index("si") for split in ("train", "test")}
    holdout = pl.read_parquet(args.holdout).join(countries["train"], on="si")
    test = pl.read_parquet(args.test).join(countries["test"], on="si")
    if args.mode == "odds":
        fits = fit(holdout, test, args.column, args.min_p, args.bins)
        adjusted = apply(test, args.column, fits, args.min_p)
    else:
        fits = per_bin(holdout, test, args.column, args.min_p, args.bins)
        adjusted = apply_per_bin(test, args.column, fits, args.min_p)
    adjusted.drop("country").write_parquet(args.output)
    args.output.with_suffix(".json").write_text(json.dumps(fits, indent=1))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
