"""Union key-based and embedding candidates, then add embedding context features.

Steps (per split):
  candidates: union of the key blocker's combined candidates and the encoder's
              top-k neighbours, stored as DuckDB table {split}_v3_candidates.
  features:   string features from features.py plus embedding features, written
              as parquet partitions under cache/v3/{split}/.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import torch

from .blocking import connect
from .features import create_features


EMBED_FEATURES = (
    "cosine", "embed_rank", "in_keys", "in_knn", "cos_minus_top", "cos_rank_union",
    "s_top_cos", "s_tenth_cos", "t_best_cos", "t_second_cos", "s_rank_for_t",
    "cos_minus_t_best", "sib_max", "sib_support", "sib_count", "union_size",
    "name_idf_cos", "name_idf_max_shared", "name_idf_s_cover", "name_idf_t_cover",
    "name_idf_max_missing", "address_idf_cos", "address_idf_s_cover", "address_idf_t_cover",
    "s_name_dup", "t_name_dup", "s_name_in_targets",
    "legal_both", "legal_equal", "legal_conflict",
)

LEGAL_FORMS = {
    "inc": "inc", "incorporated": "inc", "llc": "llc", "ltd": "ltd", "limited": "ltd",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co", "lp": "lp",
    "llp": "llp", "plc": "plc", "pc": "pc", "pllc": "pllc", "pvt": "pvt", "private": "pvt",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
}


def legal_forms(name: pl.Expr) -> pl.Expr:
    """Canonical legal-form set; dotted forms like s a s are joined first."""
    joined = name.str.replace_all(r"\b([a-z]) (?:([a-z]) )?(?:([a-z]) )?([a-z])\b", "$1$2$3$4")
    return (joined.str.split(" ").list.eval(pl.element().replace_strict(LEGAL_FORMS, default=None))
            .list.drop_nulls().list.unique().list.sort().list.join(","))



def token_tables(cache_dir: Path, split: str) -> tuple[dict, dict]:
    """Hashed token lists with IDF weights over all records of a split.

    Source 1 rows are keyed by si and target rows by ti (file row order, the
    same order as the embedding id tables) to keep memory small.
    """
    def read(i: int) -> pl.DataFrame:
        return pl.read_csv(cache_dir / f"{split}_source{i}.tsv", separator="\t", quote_char=None,
                           infer_schema=False, missing_utf8_is_empty_string=True,
                           columns=["country", "name", "name_core", "address_core"])
    source1 = read(1).with_row_index("key")
    targets = pl.concat([read(2), read(3)]).with_row_index("key")
    total = source1.height + targets.height
    frames: dict = {}
    for field in ("name", "address"):
        def explode(frame: pl.DataFrame) -> pl.DataFrame:
            return (frame.select("key", pl.col(f"{field}_core").str.split(" ").list.unique().alias("tok"))
                    .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
                    .select("key", pl.col("tok").hash().alias("tok")))
        s_tok, t_tok = explode(source1), explode(targets)
        idf = (pl.concat([s_tok.select("tok"), t_tok.select("tok")]).group_by("tok").agg(pl.len().alias("df"))
               .select("tok", (np.log(total) - pl.col("df").log()).cast(pl.Float32).alias("idf")))
        frames[field] = (s_tok.join(idf, on="tok"), t_tok.join(idf, on="tok"))
        del s_tok, t_tok, idf
    s_names = source1.filter(pl.col("name_core") != "").group_by("country", "name_core").agg(pl.len().alias("n"))
    t_names = targets.filter(pl.col("name_core") != "").group_by("country", "name_core").agg(pl.len().alias("n"))
    forms = lambda frame: legal_forms(pl.col("name")).alias("forms")
    counts = {
        "s": source1.join(s_names, on=["country", "name_core"], how="left")
                    .join(t_names.rename({"n": "m"}), on=["country", "name_core"], how="left")
                    .select(pl.col("key").alias("si"), pl.col("n").fill_null(0).cast(pl.Float32).alias("s_name_dup"),
                            pl.col("m").fill_null(0).cast(pl.Float32).alias("s_name_in_targets"),
                            forms(source1).alias("s_forms")),
        "t": targets.join(t_names, on=["country", "name_core"], how="left")
                    .select(pl.col("key").alias("ti"), pl.col("n").fill_null(0).cast(pl.Float32).alias("t_name_dup"),
                            forms(targets).alias("t_forms")),
    }
    return frames, counts


def idf_features(pairs: pl.DataFrame, frames: dict, counts: dict) -> pl.DataFrame:
    pairs = pairs.with_row_index("pair")
    ids = pairs.select("pair", "si", "ti")
    for field in ("name", "address"):
        s_tokens, t_tokens = frames[field]
        s_tok = ids.join(s_tokens, left_on="si", right_on="key").select("pair", "tok", "idf")
        t_tok = ids.join(t_tokens, left_on="ti", right_on="key").select("pair", "tok", "idf")
        s_sum = s_tok.group_by("pair").agg((pl.col("idf") ** 2).sum().alias("ss"), pl.col("idf").sum().alias("s1"))
        t_sum = t_tok.group_by("pair").agg((pl.col("idf") ** 2).sum().alias("ts"), pl.col("idf").sum().alias("t1"))
        shared = s_tok.join(t_tok.select("pair", "tok"), on=["pair", "tok"]).group_by("pair").agg(
            (pl.col("idf") ** 2).sum().alias("sh2"), pl.col("idf").sum().alias("sh1"),
            pl.col("idf").max().alias("shmax"))
        missing = s_tok.join(t_tok.select("pair", "tok"), on=["pair", "tok"], how="anti").group_by("pair").agg(
            pl.col("idf").max().alias("mmax"))
        stats = (ids.select("pair").join(s_sum, on="pair", how="left").join(t_sum, on="pair", how="left")
                 .join(shared, on="pair", how="left").join(missing, on="pair", how="left").fill_null(0))
        stats = stats.select(
            "pair",
            (pl.col("sh2") / (pl.col("ss") * pl.col("ts")).sqrt().clip(1e-6)).alias(f"{field}_idf_cos"),
            (pl.col("sh1") / pl.col("s1").clip(1e-6)).alias(f"{field}_idf_s_cover"),
            (pl.col("sh1") / pl.col("t1").clip(1e-6)).alias(f"{field}_idf_t_cover"),
            pl.col("shmax").alias(f"{field}_idf_max_shared"),
            pl.col("mmax").alias(f"{field}_idf_max_missing"),
        )
        if field == "address":
            stats = stats.drop("address_idf_max_shared", "address_idf_max_missing")
        pairs = pairs.join(stats, on="pair", how="left")
    pairs = pairs.join(counts["s"], on="si", how="left").join(counts["t"], on="ti", how="left")
    both = (pl.col("s_forms") != "") & (pl.col("t_forms") != "")
    pairs = pairs.with_columns(
        both.cast(pl.Float32).alias("legal_both"),
        (both & (pl.col("s_forms") == pl.col("t_forms"))).cast(pl.Float32).alias("legal_equal"),
        (both & (pl.col("s_forms") != pl.col("t_forms"))).cast(pl.Float32).alias("legal_conflict"),
    ).drop("s_forms", "t_forms")
    return pairs.sort("pair").drop("pair").with_columns(pl.col(pl.Float64).cast(pl.Float32))



def id_frames(folder: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    s = pl.read_parquet(folder / f"{split}_source1_ids.parquet").select("entity_id").with_row_index("si")
    t = pl.concat([pl.read_parquet(folder / f"{split}_source{i}_ids.parquet").select("entity_id")
                   for i in (2, 3)]).with_row_index("ti")
    return s, t


def build_candidates(args: argparse.Namespace) -> None:
    folder = Path(args.embed_dir)
    s_ids, t_ids = id_frames(folder, args.split)
    knn = (
        pl.scan_parquet(folder / f"{args.split}_knn.parquet")
        .filter(pl.col("rank") <= args.k)
        .join(s_ids.lazy(), left_on="q", right_on="si")
        .rename({"entity_id": "source1_entity_id"})
        .join(t_ids.lazy(), left_on="d", right_on="ti")
        .rename({"entity_id": "candidate_entity_id"})
        .select("source1_entity_id", "candidate_entity_id")
        .collect()
    )
    print(f"knn pairs: {knn.height:,}", flush=True)
    connection = connect(Path(args.database), Path(args.cache_dir))
    connection.register("knn_frame", knn.to_arrow())
    base = args.base_table
    assert base.replace("_", "").isalnum()
    out = f"{args.split}_v3_candidates"
    source_filter = "TRUE"
    if args.split == "train":
        source_filter = f"hash(source1_entity_id) % 1000 < {int(args.sample)}"
    connection.execute(f"""
        CREATE OR REPLACE TABLE {out} AS
        WITH k AS (SELECT * FROM knn_frame WHERE {source_filter}),
        b AS (SELECT * FROM {base} WHERE {source_filter}),
        u AS (
            SELECT coalesce(b.source1_entity_id, k.source1_entity_id) AS source1_entity_id,
                   coalesce(b.candidate_entity_id, k.candidate_entity_id) AS candidate_entity_id,
                   coalesce(b.block_weight, 0) AS block_weight,
                   coalesce(b.matching_keys, 0) AS matching_keys,
                   b.name_jw, b.address_jw,
                   coalesce(b.candidate_rank, 200) AS candidate_rank,
                   (b.source1_entity_id IS NOT NULL)::INT AS in_keys,
                   (k.source1_entity_id IS NOT NULL)::INT AS in_knn
            FROM b FULL OUTER JOIN k
              ON b.source1_entity_id = k.source1_entity_id
             AND b.candidate_entity_id = k.candidate_entity_id
        )
        SELECT u.* REPLACE (
            coalesce(u.name_jw, jaro_winkler_similarity(s.name, t.name)) AS name_jw,
            coalesce(u.address_jw, CASE WHEN s.address <> '' AND t.address <> ''
                     THEN jaro_winkler_similarity(s.address, t.address) ELSE 0 END) AS address_jw)
        FROM u
        JOIN {args.split}_source1 s ON u.source1_entity_id = s.entity_id
        JOIN {args.split}_targets t ON u.candidate_entity_id = t.entity_id
    """)
    stats = connection.execute(
        f"SELECT count(*), sum(in_keys), sum(in_knn), count(DISTINCT source1_entity_id) FROM {out}"
    ).fetchone()
    print(f"{out}: {stats[0]:,} pairs, keys {stats[1]:,}, knn {stats[2]:,}, sources {stats[3]:,}", flush=True)


@torch.inference_mode()
def context_features(pairs: pl.DataFrame, s_vec: np.ndarray, t_vec: np.ndarray,
                     max_group: int = 160, batch_sources: int = 2048) -> pl.DataFrame:
    """Cosine and within-Source-1 sibling features. `pairs` must be sorted by si."""
    si = pairs["si"].to_numpy()
    ti = pairs["ti"].to_numpy()
    starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    ends = np.r_[starts[1:], len(si)]
    n = len(si)
    cosine = np.zeros(n, np.float32)
    rank = np.zeros(n, np.float32)
    top = np.zeros(n, np.float32)
    sib_max = np.zeros(n, np.float32)
    sib_support = np.zeros(n, np.float32)
    sib_count = np.zeros(n, np.float32)
    size = np.repeat((ends - starts).astype(np.float32), ends - starts)
    for b in range(0, len(starts), batch_sources):
        group_starts = starts[b:b + batch_sources]
        group_ends = ends[b:b + batch_sources]
        lengths = np.minimum(group_ends - group_starts, max_group)
        g = int(lengths.max())
        offsets = np.arange(g)
        valid = offsets[None, :] < lengths[:, None]
        rows = np.where(valid, group_starts[:, None] + offsets[None, :], 0)
        targets = torch.from_numpy(np.ascontiguousarray(t_vec[ti[rows]])).cuda().float()
        sources = torch.from_numpy(np.ascontiguousarray(s_vec[si[group_starts]])).cuda().float()
        mask = torch.from_numpy(valid).cuda()
        cs = torch.einsum("bgd,bd->bg", targets, sources)
        cs_masked = cs.masked_fill(~mask, -2)
        order = torch.argsort(torch.argsort(cs_masked, dim=1, descending=True), dim=1) + 1
        tt = targets @ targets.transpose(1, 2)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask &= ~torch.eye(g, dtype=torch.bool, device="cuda")[None]
        tt = tt.masked_fill(~pair_mask, -2)
        support = torch.minimum(tt, cs_masked[:, None, :])
        count = ((tt > 0.85) & (cs_masked[:, None, :] > 0.85)).sum(2)
        flat = rows[valid]
        cosine[flat] = cs[mask].cpu().numpy()
        rank[flat] = order[mask].float().cpu().numpy()
        top[flat] = np.repeat(cs_masked.max(1).values.cpu().numpy(), lengths)
        sib_max[flat] = tt.max(2).values[mask].cpu().numpy()
        sib_support[flat] = support.max(2).values[mask].cpu().numpy()
        sib_count[flat] = count[mask].float().cpu().numpy()
    # Rows beyond max_group (rare) get exact cosine only.
    overflow = np.flatnonzero(np.repeat(ends - starts > max_group, ends - starts)
                              & (np.arange(n) - np.repeat(starts, ends - starts) >= max_group))
    if len(overflow):
        cosine[overflow] = np.einsum("nd,nd->n", s_vec[si[overflow]].astype(np.float32),
                                     t_vec[ti[overflow]].astype(np.float32))
        rank[overflow] = max_group + 1
        sib_max[overflow] = sib_support[overflow] = -2
    return pairs.with_columns(
        cosine=cosine, cos_rank_union=rank, cos_minus_top=cosine - top,
        sib_max=sib_max, sib_support=sib_support, sib_count=sib_count, union_size=size,
    )


def build_features(args: argparse.Namespace) -> None:
    folder = Path(args.embed_dir)
    out_dir = Path(args.cache_dir) / "v3" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("part*.parquet"):
        stale.unlink()
    connection = connect(Path(args.database), Path(args.cache_dir))
    candidates = f"{args.split}_v3_candidates"
    features = f"{args.split}_v3_features"
    if not args.reuse_features:
        started = time.time()
        create_features(connection, args.split, candidates, features)
        connection.execute(f"""
            CREATE OR REPLACE TABLE {features} AS
            SELECT f.*, c.in_keys::FLOAT AS in_keys, c.in_knn::FLOAT AS in_knn
            FROM {features} f JOIN {candidates} c USING (source1_entity_id, candidate_entity_id)
        """)
        print(f"string features in {time.time() - started:.0f}s", flush=True)
    # Export partitions, then release DuckDB's buffer pool before the polars stage.
    raw_dir = Path(args.cache_dir) / "v3" / f"raw_{args.split}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for part in range(args.partitions):
        connection.execute(f"""
            COPY (SELECT *, (hash(source1_entity_id) % 1000)::SMALLINT AS bucket FROM {features}
                  WHERE hash(source1_entity_id) % {args.partitions} = {part})
            TO '{raw_dir / f"raw{part:02d}.parquet"}' (FORMAT parquet)
        """)
    connection.close()
    print("exported raw partitions", flush=True)
    s_ids, t_ids = id_frames(folder, args.split)
    s_vec = np.load(folder / f"{args.split}_source1.npy", mmap_mode="r")
    t_vec = np.concatenate([np.load(folder / f"{args.split}_source{i}.npy", mmap_mode="r") for i in (2, 3)])
    knn = pl.scan_parquet(folder / f"{args.split}_knn.parquet")
    s_stats = knn.group_by("q").agg(
        pl.col("cosine").filter(pl.col("rank") == 1).first().alias("s_top_cos"),
        pl.col("cosine").filter(pl.col("rank") == 10).first().alias("s_tenth_cos"),
    ).rename({"q": "si"}).collect()
    rknn = pl.scan_parquet(folder / f"{args.split}_rknn.parquet")
    t_stats = rknn.group_by("q").agg(
        pl.col("cosine").filter(pl.col("rank") == 1).first().alias("t_best_cos"),
        pl.col("cosine").filter(pl.col("rank") == 2).first().alias("t_second_cos"),
    ).rename({"q": "ti"}).collect()
    s_rank = rknn.select(pl.col("q").alias("ti"), pl.col("d").alias("si"),
                         pl.col("rank").cast(pl.Float32).alias("s_rank_for_t"))
    knn_rank = knn.filter(pl.col("rank") <= args.k).select(
        pl.col("q").alias("si"), pl.col("d").alias("ti"), pl.col("rank").cast(pl.Float32).alias("embed_rank"))
    frames, counts = token_tables(Path(args.cache_dir), args.split)
    print("token tables ready", flush=True)
    for part in range(args.partitions):
        started = time.time()
        raw = raw_dir / f"raw{part:02d}.parquet"
        frame = (pl.read_parquet(raw)
                 .join(s_ids, left_on="source1_entity_id", right_on="entity_id")
                 .join(t_ids, left_on="candidate_entity_id", right_on="entity_id"))
        sources = frame.select("si").unique()
        # Restrict the large neighbour tables to this partition's entities before joining.
        part_rank = s_rank.join(sources.lazy(), on="si", how="semi").collect()
        part_knn = knn_rank.join(sources.lazy(), on="si", how="semi").collect()
        frame = (
            frame.join(s_stats, on="si", how="left")
            .join(t_stats, on="ti", how="left")
            .join(part_rank, on=["ti", "si"], how="left")
            .join(part_knn, on=["si", "ti"], how="left")
            .with_columns(
                pl.col("s_rank_for_t").fill_null(6.0),
                pl.col("embed_rank").fill_null(args.k + 1.0),
            )
            .sort("si")
        )
        del part_rank, part_knn
        frame = context_features(frame, s_vec, t_vec)
        frame = idf_features(frame, frames, counts)
        frame = frame.with_columns((pl.col("cosine") - pl.col("t_best_cos")).alias("cos_minus_t_best"))
        frame.write_parquet(out_dir / f"part{part:02d}.parquet")
        raw.unlink()
        print(f"partition {part + 1}/{args.partitions}: {frame.height:,} rows in {time.time() - started:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("candidates", "features"))
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--database", default="cache/resolution.duckdb")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--embed-dir", default="cache/embeddings")
    parser.add_argument("--base-table")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--partitions", type=int, default=1)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.base_table is None:
        args.base_table = ("train_sample_candidates" if args.split == "train"
                           else "test_candidates")
    {"candidates": build_candidates, "features": build_features}[args.command](args)


if __name__ == "__main__":
    main()
