"""Monte Carlo estimate of macro F0.5 from calibrated pair posteriors (no labels).

Labels are sampled independently from each pair's posterior; the chosen matches
are scored against each sample and averaged. On the labelled holdout the
estimate can be compared with the true score to check its bias.
"""
import sys
import numpy as np
import polars as pl
from pathlib import Path
from amzn.calibrate import per_bin, apply_per_bin, per_segment, apply_per_segment, house_segments
from amzn.train_v4 import one_to_one
from amzn.train_v3 import decode


def estimate(frame: pl.DataFrame, column: str, chosen: dict, entities: int, samples: int = 3, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    si = frame["si"].to_numpy()
    ti = frame["ti"].to_numpy()
    p = frame[column].to_numpy().astype(np.float64)
    picked = set((k, t) for k, v in chosen.items() for t in v)
    is_chosen = np.fromiter(((a, b) in picked for a, b in zip(si.tolist(), ti.tolist())), bool, len(si))
    scores = []
    for _ in range(samples):
        y = rng.random(len(p)) < p
        g = pl.DataFrame({"si": si, "y": y, "c": is_chosen, "tp": y & is_chosen}).group_by("si").agg(
            pl.col("y").sum().alias("a"), pl.col("c").sum().alias("k"), pl.col("tp").sum().alias("tp"))
        a, k, tp = g["a"].to_numpy(), g["k"].to_numpy(), g["tp"].to_numpy()
        f = np.where((a == 0) & (k == 0), 1.0,
                     np.where((a == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * a + k, 1e-9)))
        # Entities without any pair: both empty -> 1.
        scores.append((f.sum() + (entities - len(g))) / entities)
    return float(np.mean(scores))


def run(model_dir: str, column: str, calibrate: bool, label: str, test_file: str = "test_scored.parquet",
        segments: bool = False) -> None:
    c = {s: pl.read_parquet(f"cache/embeddings/{s}_source1_ids.parquet").select("country").with_row_index("si")
         for s in ("train", "test")}
    hold = pl.read_parquet(f"{model_dir}/valid_predictions.parquet").join(c["train"], on="si")
    test = pl.read_parquet(f"{model_dir}/{test_file}").join(c["test"], on="si")
    test = test.filter(pl.col(column) >= 0.001)
    if calibrate and segments:
        hold_s = house_segments(hold, Path("cache"), "train")
        test = house_segments(test, Path("cache"), "test")
        fits = per_segment(hold_s, test, column, 0.02, 25)
        test = apply_per_segment(test, column, fits, 0.02)
    elif calibrate:
        fits = per_bin(hold, test, column, 0.02, 25)
        test = apply_per_bin(test, column, fits, 0.02)
    scored = one_to_one(test.select("si", "ti", column), column)
    chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored[column].to_numpy(), "expected", 0.05, 0.0)
    est = estimate(scored, column, chosen, 1_732_544)
    # Holdout check: raw holdout scores are calibrated for the holdout itself.
    h = one_to_one(hold.select("si", "ti", column, "label"), column)
    hc = decode(h["si"].to_numpy(), h["ti"].to_numpy(), h[column].to_numpy(), "expected", 0.05, 0.0)
    n_hold = 22144
    est_h = estimate(h, column, hc, n_hold)
    picked = set((k, t) for k, v in hc.items() for t in v)
    true = h.with_columns(pl.Series("c", [(a, b) in picked for a, b in zip(h["si"].to_list(), h["ti"].to_list())]))
    g = true.group_by("si").agg(pl.col("label").sum().alias("a"), pl.col("c").sum().alias("k"),
                                (pl.col("label").cast(pl.Boolean) & pl.col("c")).sum().alias("tp"))
    a, k, tp = g["a"].to_numpy(), g["k"].to_numpy(), g["tp"].to_numpy()
    f = np.where((a == 0) & (k == 0), 1.0, np.where((a == 0) | (k == 0), 0.0, 1.25 * tp / np.maximum(0.25 * a + k, 1e-9)))
    print(f"{label}: estimated test {est:.5f} | holdout estimated {est_h:.5f} vs actual-in-candidates "
          f"{(f.sum() + n_hold - len(g)) / n_hold:.5f}", flush=True)


if __name__ == "__main__":
    jobs = {
        "v3": ("output/v3_legal", "p2", True, "v3 p2 calibrated (LB 0.968379)", "test_scored_pairs.parquet"),
        "v3raw": ("output/v3_legal", "p2", False, "v3 p2 raw (LB 0.960932)", "test_scored_pairs.parquet"),
        "v4p2": ("output/v4", "p2", True, "v4 p2 calibrated", "test_scored.parquet"),
        "v4p1": ("output/v4", "p1", True, "v4 p1 calibrated", "test_scored.parquet"),
        "v3seg": ("output/v3_legal", "p2", True, "v3 p2 segmented", "test_scored_pairs.parquet", True),
        "v4p2seg": ("output/v4", "p2", True, "v4 p2 segmented", "test_scored.parquet", True),
        "v4p1seg": ("output/v4", "p1", True, "v4 p1 segmented", "test_scored.parquet", True),
        "househ": ("output/blend_house", "p2", True, "blend + house relation", "test_scored.parquet", True),
        "housett": ("output/blend_house_tt", "p2", True, "blend + house + target pairs", "test_scored.parquet", True),
        "blendseg": ("output/v4_blend", "p2", True, "v4 p1+CE blend segmented", "test_scored.parquet", True),
    }
    for name in sys.argv[1:] or jobs:
        run(*jobs[name])
