"""Fine-tune a multilingual bi-encoder, encode records, and retrieve nearest targets.

The encoder is intfloat/multilingual-e5-small (MIT license, 118M parameters).
Training uses only the supplied training ground truth. Source 1 records whose
DuckDB hash bucket is below `--min-bucket` are excluded so that the pair model
training and validation entities never influence the encoder.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


BASE_MODEL = "intfloat/multilingual-e5-small"
MAX_LENGTH = 80


def read_source(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True,
    )


def clean(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value.lower() in {"none", "null", "nan"} else value


def record_text(name: str | None, address: str | None) -> str:
    return f"query: {clean(name)} | {clean(address)}"


def texts_for(frame: pl.DataFrame) -> list[str]:
    return [record_text(n, a) for n, a in zip(frame["business_name"], frame["business_address"])]


def mean_pool(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(output.dtype)
    return (output * mask).sum(1) / mask.sum(1).clamp(min=1)


class Encoder(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(**batch).last_hidden_state
        return F.normalize(mean_pool(output, batch["attention_mask"]), dim=-1)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    random.seed(0)
    data = Path(args.data_dir)
    source1 = read_source(data / "train_source1.tsv")
    targets = pl.concat([read_source(data / "train_source2.tsv"), read_source(data / "train_source3.tsv")])
    truth = read_source(data / "train_ground_truth.tsv")
    buckets = duckdb.sql(
        "SELECT entity_id, hash(entity_id) % 1000 AS bucket FROM source1"
    ).pl()
    allowed = buckets.filter(pl.col("bucket") >= args.min_bucket)["entity_id"]
    pairs = (
        truth.filter(pl.col("source1_entity_id").is_in(allowed.implode()))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids") != "")
        .rename({"source1_entity_id": "s", "matched_entity_ids": "t"})
        .sample(n=args.pairs, seed=0)
    )
    left = source1.select(
        pl.col("entity_id").alias("s"), "country",
        pl.struct("business_name", "business_address").alias("s_record"),
    )
    right = targets.select(
        pl.col("entity_id").alias("t"),
        pl.struct("business_name", "business_address").alias("t_record"),
    )
    pairs = pairs.join(left, on="s").join(right, on="t")
    print(f"Training pairs: {pairs.height:,}", flush=True)

    # Hard in-batch negatives: half the batches group Source 1 records sharing a
    # name prefix, the other half records sharing trailing address tokens (city).
    names = pairs["s_record"].struct.field("business_name").fill_null("").str.to_lowercase()
    addresses = pairs["s_record"].struct.field("business_address").fill_null("").str.to_lowercase()
    name_key = names.str.replace_all(r"[^\p{L}\p{N}]", "").str.slice(0, 5)
    city_key = addresses.str.split(",").list.slice(-2, 2).list.join(",").str.strip_chars()
    rng = np.random.default_rng(0)
    frame = pl.DataFrame({
        "i": np.arange(pairs.height), "country": pairs["country"],
        "key": pl.Series(np.where(rng.random(pairs.height) < 0.5,
                                  ("n" + name_key).to_numpy(), ("a" + city_key).to_numpy())),
        "noise": rng.random(pairs.height),
    }).sort("country", "key", "noise")
    order = frame["i"].to_list()
    batches = [order[s:s + args.batch_size]
               for s in range(0, len(order) - args.batch_size + 1, args.batch_size)]
    random.shuffle(batches)
    left_text = [record_text(r["business_name"], r["business_address"]) for r in pairs["s_record"]]
    right_text = [record_text(r["business_name"], r["business_address"]) for r in pairs["t_record"]]
    left_ids = pairs["s"].to_list()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = Encoder(BASE_MODEL).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(batches)
    warmup = max(1, total // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * step / total))),
    )
    scale = 1.0 / args.temperature
    model.train()
    started = time.time()
    for step, batch in enumerate(batches):
        a = tokenizer([left_text[i] for i in batch], padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        b = tokenizer([right_text[i] for i in batch], padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ea = model(a)
            eb = model(b)
        logits = (ea.float() @ eb.float().T) * scale
        # Several targets of one Source 1 entity can share a batch; mask them as non-negatives.
        codes = torch.tensor(pl.Series([left_ids[i] for i in batch]).rank("dense").to_numpy(), device="cuda")
        same = codes[:, None] == codes[None, :]
        eye = torch.eye(len(batch), dtype=torch.bool, device="cuda")
        logits = logits.masked_fill(same & ~eye, -1e4)
        labels = torch.arange(len(batch), device="cuda")
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            rate = (step + 1) * args.batch_size / (time.time() - started)
            print(f"step {step}/{total} loss {loss.item():.4f} pairs/s {rate:.0f}", flush=True)
    output = Path(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    print(f"Saved encoder to {output}", flush=True)


@torch.inference_mode()
def encode_texts(model: Encoder, tokenizer, texts: list[str], batch_size: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts])
    result = np.empty((len(texts), model.model.config.hidden_size), dtype=np.float16)
    started = time.time()
    for n, start in enumerate(range(0, len(texts), batch_size)):
        index = order[start:start + batch_size]
        batch = tokenizer([texts[i] for i in index], padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            result[index] = model(batch).half().cpu().numpy()
        if n % 500 == 0:
            print(f"  encoded {start:,}/{len(texts):,} ({start / max(time.time() - started, 1e-9):.0f}/s)", flush=True)
    return result


def encode(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = Encoder(args.model_dir).cuda().eval()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for source in (1, 2, 3):
        frame = read_source(Path(args.data_dir) / f"{args.split}_source{source}.tsv")
        vectors = encode_texts(model, tokenizer, texts_for(frame), args.batch_size)
        np.save(out / f"{args.split}_source{source}.npy", vectors)
        frame.select("entity_id", "country").write_parquet(out / f"{args.split}_source{source}_ids.parquet")
        print(f"Encoded {args.split} source {source}: {len(frame):,}", flush=True)


def target_vectors(folder: Path, split: str) -> np.ndarray:
    """Memory-mapped Source 2 + Source 3 embeddings in target (ti) order."""
    path = folder / f"{split}_targets.npy"
    if not path.exists():
        parts = [np.load(folder / f"{split}_source{i}.npy", mmap_mode="r") for i in (2, 3)]
        combined = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=parts[0].dtype,
                                             shape=(sum(len(p) for p in parts), parts[0].shape[1]))
        offset = 0
        for part in parts:
            for start in range(0, len(part), 1 << 20):
                chunk = part[start:start + (1 << 20)]
                combined[offset + start:offset + start + len(chunk)] = chunk
            offset += len(part)
        combined.flush()
        del combined
        path.with_suffix(".tmp.npy").rename(path)
    return np.load(path, mmap_mode="r")


def load_side(folder: Path, split: str, sources: tuple[int, ...]) -> tuple[pl.DataFrame, np.ndarray]:
    ids = pl.concat([pl.read_parquet(folder / f"{split}_source{i}_ids.parquet") for i in sources])
    vectors = np.concatenate([np.load(folder / f"{split}_source{i}.npy", mmap_mode="r") for i in sources])
    return ids, vectors


@torch.inference_mode()
def topk_chunked(query: torch.Tensor, database: torch.Tensor, k: int, chunk: int = 1 << 20):
    best_scores = best_index = None
    for start in range(0, database.shape[0], chunk):
        scores, index = torch.topk(query @ database[start:start + chunk].T,
                                   min(k, database.shape[0] - start), dim=1)
        index += start
        if best_scores is not None:
            scores = torch.cat([best_scores, scores], 1)
            index = torch.cat([best_index, index], 1)
            scores, order = torch.topk(scores, k, dim=1)
            index = torch.gather(index, 1, order)
        best_scores, best_index = scores, index
    return best_scores, best_index


@torch.inference_mode()
def neighbours(args: argparse.Namespace) -> None:
    """Exact cosine top-k per country on GPU; country is an open label set.

    Forward: Source 1 queries against Source 2+3. Reverse: Source 2+3 queries
    against Source 1 (competition features). Output rows hold integer positions
    into the concatenated *_ids.parquet order of each side.
    """
    folder = Path(args.output_dir)
    s_ids, s_vec = load_side(folder, args.split, (1,))
    t_ids, t_vec = load_side(folder, args.split, (2, 3))
    q_ids, q_vec, d_ids, d_vec = (t_ids, t_vec, s_ids, s_vec) if args.reverse else (s_ids, s_vec, t_ids, t_vec)
    keep = np.ones(len(q_ids), dtype=bool)
    if args.source_filter_buckets is not None and not args.reverse:
        keep = (duckdb.sql("SELECT hash(entity_id) % 1000 AS b FROM q_ids").pl()["b"]
                < args.source_filter_buckets).to_numpy()
    q_parts, d_parts, score_parts = [], [], []
    for country in sorted(set(q_ids["country"].to_list())):
        q_index = np.flatnonzero((q_ids["country"] == country).to_numpy() & keep)
        d_index = np.flatnonzero((d_ids["country"] == country).to_numpy())
        if len(q_index) == 0 or len(d_index) == 0:
            continue
        database = torch.from_numpy(np.ascontiguousarray(d_vec[d_index])).cuda()
        k = min(args.k, len(d_index))
        started = time.time()
        for start in range(0, len(q_index), args.query_batch):
            rows = q_index[start:start + args.query_batch]
            query = torch.from_numpy(np.ascontiguousarray(q_vec[rows])).cuda()
            scores, index = topk_chunked(query, database, k)
            q_parts.append(np.repeat(rows.astype(np.int32), k))
            d_parts.append(d_index[index.cpu().numpy()].astype(np.int32).ravel())
            score_parts.append(scores.float().cpu().numpy().ravel())
        print(f"{country}: {len(q_index):,} queries x {len(d_index):,} in {time.time() - started:.0f}s", flush=True)
        del database
        torch.cuda.empty_cache()
    query_rows = np.concatenate(q_parts)
    result = pl.DataFrame({
        "q": query_rows, "d": np.concatenate(d_parts), "cosine": np.concatenate(score_parts),
    }).with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.Int16).over("q").alias("rank"))
    name = f"{args.split}_{'rknn' if args.reverse else 'knn'}.parquet"
    result.write_parquet(folder / name)
    print(f"Wrote {result.height:,} neighbour pairs to {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data-dir", default="dataset/train")
    p.add_argument("--model-dir", default="models/encoder")
    p.add_argument("--pairs", type=int, default=1_500_000)
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--min-bucket", type=int, default=100)
    p = sub.add_parser("encode")
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--data-dir")
    p.add_argument("--model-dir", default="models/encoder")
    p.add_argument("--output-dir", default="cache/embeddings")
    p.add_argument("--batch-size", type=int, default=1024)
    p = sub.add_parser("knn")
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--output-dir", default="cache/embeddings")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--query-batch", type=int, default=2048)
    p.add_argument("--reverse", action="store_true")
    p.add_argument("--source-filter-buckets", type=int)
    args = parser.parse_args()
    if getattr(args, "data_dir", None) is None:
        args.data_dir = f"dataset/{args.split}"
    {"train": train, "encode": encode, "knn": neighbours}[args.command](args)


if __name__ == "__main__":
    main()
