"""Generator-aware name features for candidate pairs.

The data generator perturbs true records and plants decoys in recognisable ways.
A true record keeps the Source 1 name up to typos, dropped words, appended
suffixes, a word replaced by a stock filler (center, services, partners; France:
services, france), an acronym, a web domain or a DBA alias. A decoy swaps one
word in place for another common vocabulary word (in France "Club" becomes
"Comite", a few doors away or at the same number), or carries an unrelated
made-up brand. In training, a same-address swap to a common word is true only
45% of the time, against 96-99% for a filler word or a garbled typo.

The features describe how the target name differs from the Source 1 name, with
word frequencies counted on the Source 1 names of the pair's own country.
Pairs are those with stage-1 p1 >= --min-p (training: blend buckets < 100 only).
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from .france_rule import consistent, kind
from .v3 import id_frames

FILLERS = {"center", "services", "service", "partners", "france"}
COMMON = 20
FEATURES = ["g_exact", "g_add", "g_drop", "g_typo1", "g_swap_inplace", "g_swap_moved", "g_swap_filler",
            "g_swap_common", "g_swap_rare", "g_multi", "g_n_miss", "g_n_extra", "g_miss_df", "g_extra_df",
            "g_extra_df_min", "g_extra_filler", "g_acronym", "g_domain", "g_alias", "g_consistent", "g_brand1",
            "g_s_ntok", "g_t_ntok"]
DF: dict[str, Counter] = {}


def read(path: Path, columns: list[str]) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, columns=columns,
                       missing_utf8_is_empty_string=True)


def document_frequency(cache_dir: Path, split: str) -> dict[str, Counter]:
    frame = read(cache_dir / f"{split}_source1.tsv", ["country", "name_core"])
    result = {}
    for (country,), group in frame.group_by("country"):
        counter: Counter = Counter()
        for core in group["name_core"]:
            counter.update(set(core.split()))
        result[country] = counter
    return result


def describe(row: tuple) -> tuple:
    s_core, t_core, s_raw, t_raw, country = row
    df = DF.get(country, Counter())
    s, t = s_core.split(), t_core.split()
    ss, ts = set(s), set(t)
    miss = [w for w in s if w not in ts]
    extra = [w for w in t if w not in ss]
    exact = add = drop = typo = inplace = moved = filler = common = rare = multi = 0.0
    if not miss and not extra:
        exact = 1.0
    elif not miss:
        add = 1.0
    elif not extra:
        drop = 1.0
    elif len(miss) == 1 and len(extra) == 1:
        if SequenceMatcher(None, miss[0], extra[0]).ratio() >= 0.75:
            typo = 1.0
        else:
            if len(s) == len(t) and s.index(miss[0]) == t.index(extra[0]):
                inplace = 1.0
            else:
                moved = 1.0
            word = extra[0]
            if word in FILLERS:
                filler = 1.0
            elif df.get(word, 0) >= COMMON:
                common = 1.0
            else:
                rare = 1.0
    else:
        multi = 1.0
    log_df = lambda words, pick: math.log1p(pick(df.get(w, 0) for w in words)) if words else -1.0
    k = kind(t_raw)
    agrees = float(k != "other" and consistent(k, t_raw, s_raw, s_core))
    words = re.findall(r"[A-Za-z]+", t_raw)
    brand = float(len(words) == 1 and len(words[0]) >= 6 and words[0].lower() not in ss
                  and df.get(words[0].lower(), 0) < 5)
    return (exact, add, drop, typo, inplace, moved, filler, common, rare, multi, float(len(miss)), float(len(extra)),
            log_df(miss, max), log_df(extra, max), log_df(extra, min), float(bool(extra) and set(extra) <= FILLERS),
            float(k == "acronym"), float(k == "domain"), float(k == "alias"), agrees, brand,
            float(len(s)), float(len(t)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("output/v7"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--embed-dir", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--min-p", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    split = args.split
    out_dir = args.cache_dir / "gen"
    out_dir.mkdir(parents=True, exist_ok=True)

    global DF
    DF = document_frequency(args.cache_dir, split)
    s_ids, _ = id_frames(args.embed_dir, split)
    pairs = pl.scan_parquet(str(args.model_dir / f"{split}_p1" / "*.parquet")).filter(pl.col("p1") >= args.min_p)
    if split == "train":
        buckets = duckdb.sql("SELECT si, (hash(entity_id) % 1000)::INT AS bucket FROM s_ids").pl()
        keep = buckets.filter(pl.col("bucket") < 100).select(pl.col("si").cast(pl.UInt32))
        pairs = pairs.join(keep.lazy(), on="si", how="semi")
    pairs = pairs.select("si", "ti").collect()
    print(f"{split}: {pairs.height:,} pairs with p1 >= {args.min_p}", flush=True)

    source = read(args.cache_dir / f"{split}_source1.tsv", ["country", "name_core"]).with_columns(
        read(args.data_dir / split / f"{split}_source1.tsv", ["business_name"])["business_name"].alias("raw"))
    targets = pl.concat([
        read(args.cache_dir / f"{split}_source{i}.tsv", ["name_core"]).with_columns(
            read(args.data_dir / split / f"{split}_source{i}.tsv", ["business_name"])["business_name"].alias("raw"))
        for i in (2, 3)])
    si, ti = pairs["si"].to_numpy(), pairs["ti"].to_numpy()
    rows = zip(source["name_core"].to_numpy()[si], targets["name_core"].to_numpy()[ti],
               source["raw"].to_numpy()[si], targets["raw"].to_numpy()[ti], source["country"].to_numpy()[si])
    with Pool(args.workers) as pool:
        values = pool.map(describe, rows, chunksize=20000)
    table = pl.DataFrame(np.asarray(values, dtype=np.float32), schema=FEATURES, orient="row")
    result = pl.concat([pairs, table], how="horizontal")
    result.write_parquet(out_dir / f"{split}.parquet")
    print(f"wrote {out_dir / f'{split}.parquet'}: {result.height:,} rows", flush=True)
    print(result.select(FEATURES[:10]).mean(), flush=True)


if __name__ == "__main__":
    main()
