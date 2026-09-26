"""Holdout check of decision rules: decode the blend holdout as the hybrid does, then drop accepted pairs
matching a rule and report macro F0.5 before/after and the precision of the dropped pairs."""
import sys
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import polars as pl

from amzn.train_v3 import decode
from amzn.train_v4 import one_to_one

R = lambda p, c: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
folder = sys.argv[1] if len(sys.argv) > 1 else "output/blend_v7_fullL"
hold = pl.read_parquet(f"{folder}/valid_predictions.parquet")
N_HOLD = 22144
scored = one_to_one(hold.select("si", "ti", "p2"), "p2")
chosen = decode(scored["si"].to_numpy(), scored["ti"].to_numpy(), scored["p2"].to_numpy(), "expected", 0.05, 0.0)
acc = pl.DataFrame({"si": np.repeat(np.fromiter(chosen.keys(), np.int64), [len(v) for v in chosen.values()]),
                    "ti": np.fromiter((t for v in chosen.values() for t in v), np.int64)}).with_columns(
    pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32))
truth = hold.filter(pl.col("label") == 1).select("si", "ti")
acc = acc.join(hold.select("si", "ti", "label"), on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))


def macro(accepted: pl.DataFrame) -> float:
    a = truth.group_by("si").len().rename({"len": "a"})
    k = accepted.group_by("si").agg(pl.len().alias("k"), pl.col("label").sum().alias("tp"))
    g = a.join(k, on="si", how="full", coalesce=True).fill_null(0)
    f = np.where((g["a"] == 0) & (g["k"] == 0), 1.0, np.where((g["a"] == 0) | (g["k"] == 0), 0.0,
                 1.25 * g["tp"].to_numpy() / np.maximum(0.25 * g["a"].to_numpy() + g["k"].to_numpy(), 1e-9)))
    return (f.sum() + N_HOLD - g.height) / N_HOLD


base = macro(acc)
print(f"holdout macro F0.5 (in candidates) {base:.5f}; accepted {acc.height:,}, precision {acc['label'].mean():.4f}")

# Pair descriptors.
s_house = R("cache/train_source1.tsv", ["house"])["house"].to_numpy()
s_core = R("cache/train_source1.tsv", ["name_core"])["name_core"].to_numpy()
s_country = R("cache/train_source1.tsv", ["country"])["country"].to_numpy()
t_norm = pl.concat([R(f"cache/train_source{i}.tsv", ["address", "name_core"]) for i in (2, 3)])
t_nums = t_norm["address"].str.extract_all(r"(?:^|[^a-z0-9])(\d+)").list.eval(
    pl.element().str.replace_all(r"[^0-9]", "").str.strip_chars_start("0"))
t_core = t_norm["name_core"].to_numpy()
df = Counter()
for core in s_core:
    df.update(set(core.split()))


def last_digit(h, ns):
    h = (h or "").lstrip("0")
    if not h.isdigit() or not ns or h in ns:
        return 0
    best = None
    for n in ns:
        if n.isdigit() and len(n) == len(h) and n[:-1] == h[:-1] and n != h:
            d = abs(int(n) - int(h))
            best = d if best is None else min(best, d)
    return 0 if best is None else best


def subsequence(short, long_):
    it = iter(long_)
    return all(ch in it for ch in short)


def swap(a_core, b_core):
    s, t = (a_core or "").split(), (b_core or "").split()
    ss, ts = set(s), set(t)
    miss = [w for w in s if w not in ts]
    extra = [w for w in t if w not in ss]
    if len(miss) != 1 or len(extra) != 1:
        return False
    a, b = miss[0], extra[0]
    if b in {"center", "services", "service", "partners", "france"} or df.get(b, 0) < 20 or SequenceMatcher(None, a, b).ratio() >= 0.75:
        return False
    return not (subsequence(b, a) or subsequence(a, b))


si, ti = acc["si"].to_numpy(), acc["ti"].to_numpy()
nums = t_nums.gather(ti).to_list()
acc = acc.with_columns(
    pl.Series("last_delta", [last_digit(h, n) for h, n in zip(s_house[si], nums)]),
    pl.Series("swap", [swap(a, b) for a, b in zip(s_core[si], t_core[ti])]),
    pl.Series("country", s_country[si]))
rules = {
    "last digit, delta 1-2": (pl.col("last_delta") >= 1) & (pl.col("last_delta") <= 2),
    "last digit, delta 3-9": pl.col("last_delta") >= 3,
    "last digit, any": pl.col("last_delta") >= 1,
    "common-word swap": pl.col("swap"),
}
for name, rule in rules.items():
    dropped = acc.filter(rule)
    kept = acc.filter(~rule)
    by = dropped.group_by("country").agg(pl.len(), pl.col("label").mean().round(3)).sort("country").rows()
    print(f"{name}: drop {dropped.height:,} (precision {dropped['label'].mean() if dropped.height else float('nan'):.3f}; {by}) "
          f"-> F0.5 {macro(kept):.5f} ({macro(kept) - base:+.5f})")
