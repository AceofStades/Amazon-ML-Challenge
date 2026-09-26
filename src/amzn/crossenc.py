"""Cross-encoder reranker: reads a Source 1 record and a candidate jointly.

Base model: intfloat/multilingual-e5-small (MIT license, 118M parameters) with a
linear head on the mean-pooled output. Training pairs are each Source 1
entity's nearest encoder neighbours (hard negatives plus true matches), drawn
only from hash buckets >= --min-bucket, so the matcher's training and holdout
entities never influence it. The score becomes a stage-2 matcher feature.
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

from .embed import BASE_MODEL, clean, mean_pool, read_source

MAX_LENGTH = 160


def record_texts(data_dir: Path, split: str) -> tuple[list[str], list[str]]:
    """Texts in si order (Source 1 file) and ti order (Source 2 then Source 3)."""
    def texts(frame: pl.DataFrame) -> list[str]:
        return [f"{clean(n)} | {clean(a)}" for n, a in zip(frame["business_name"], frame["business_address"])]
    source1 = texts(read_source(data_dir / f"{split}_source1.tsv"))
    targets = texts(read_source(data_dir / f"{split}_source2.tsv")) + texts(read_source(data_dir / f"{split}_source3.tsv"))
    return source1, targets


class CrossEncoder(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(**batch).last_hidden_state
        return self.head(mean_pool(output, batch["attention_mask"])).squeeze(-1)


def training_pairs(args: argparse.Namespace) -> pl.DataFrame:
    embed = Path(args.embed_dir)
    s_ids = pl.read_parquet(embed / "train_source1_ids.parquet").select("entity_id").with_row_index("si")
    t_ids = pl.concat([pl.read_parquet(embed / f"train_source{i}_ids.parquet").select("entity_id")
                       for i in (2, 3)]).with_row_index("ti")
    buckets = duckdb.sql("SELECT si, hash(entity_id) % 1000 AS bucket FROM s_ids").pl()
    chosen = buckets.filter(pl.col("bucket") >= args.min_bucket).sample(n=args.entities, seed=1)["si"]
    knn = (pl.scan_parquet(embed / "train_knn.parquet")
           .filter((pl.col("rank") <= args.depth) & pl.col("q").is_in(chosen.implode()))
           .select(pl.col("q").alias("si"), pl.col("d").alias("ti")).collect())
    truth = (read_source(Path(args.data_dir) / "train_ground_truth.tsv")
             .with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .join(s_ids, left_on="source1_entity_id", right_on="entity_id")
             .join(t_ids, left_on="matched_entity_ids", right_on="entity_id")
             .select("si", "ti", pl.lit(1.0, pl.Float32).alias("label")))
    # Also add every true match of the chosen entities, retrieved or not.
    positives = truth.filter(pl.col("si").is_in(chosen.implode())).select("si", "ti")
    pairs = pl.concat([knn, positives]).unique()
    pairs = pairs.join(truth, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0.0))
    print(f"cross-encoder pairs {pairs.height:,}, positive rate {pairs['label'].mean():.3f}", flush=True)
    return pairs.sample(fraction=1.0, shuffle=True, seed=2)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    pairs = training_pairs(args)
    source1, targets = record_texts(Path(args.data_dir), "train")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = CrossEncoder(BASE_MODEL).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    si, ti, labels = pairs["si"].to_numpy(), pairs["ti"].to_numpy(), pairs["label"].to_numpy()
    total = len(si) // args.batch_size
    warmup = max(1, total // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)))
    model.train()
    started = time.time()
    running = 0.0
    for step in range(total):
        rows = slice(step * args.batch_size, (step + 1) * args.batch_size)
        batch = tokenizer([source1[i] for i in si[rows]], [targets[i] for i in ti[rows]], padding=True,
                          truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        target = torch.from_numpy(labels[rows]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running = 0.98 * running + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0:
            rate = (step + 1) * args.batch_size / (time.time() - started)
            print(f"step {step}/{total} loss {running:.4f} pairs/s {rate:.0f}", flush=True)
    output = Path(args.model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    torch.save(model.head.state_dict(), output / "head.pt")
    print(f"saved cross-encoder to {output}", flush=True)


@torch.inference_mode()
def score(args: argparse.Namespace) -> None:
    """Score (si, ti) pairs from --pairs parquet; writes si, ti, ce_logit."""
    pairs = pl.read_parquet(args.pairs, columns=["si", "ti"])
    source1, targets = record_texts(Path(args.data_dir), args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = CrossEncoder(args.model_dir).cuda().eval()
    model.head.load_state_dict(torch.load(Path(args.model_dir) / "head.pt"))
    si, ti = pairs["si"].to_numpy(), pairs["ti"].to_numpy()
    lengths = np.fromiter((len(source1[a]) + len(targets[b]) for a, b in zip(si, ti)), np.int32, len(si))
    order = np.argsort(lengths, kind="stable")
    result = np.empty(len(si), np.float32)
    started = time.time()
    for n, start in enumerate(range(0, len(order), args.batch_size)):
        index = order[start:start + args.batch_size]
        batch = tokenizer([source1[i] for i in si[index]], [targets[i] for i in ti[index]], padding=True,
                          truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            result[index] = model(batch).float().cpu().numpy()
        if n % 500 == 0:
            print(f"  scored {start:,}/{len(order):,} ({start / max(time.time() - started, 1e-9):.0f}/s)", flush=True)
    pairs.with_columns(ce_logit=result).write_parquet(args.output)
    print(f"wrote {args.output}: {len(result):,} pairs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--data-dir", default="dataset/train")
    p.add_argument("--embed-dir", default="cache/embeddings")
    p.add_argument("--model-dir", default="models/crossencoder")
    p.add_argument("--entities", type=int, default=400_000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--min-bucket", type=int, default=100)
    p = sub.add_parser("score")
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--data-dir")
    p.add_argument("--model-dir", default="models/crossencoder")
    p.add_argument("--pairs", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if getattr(args, "data_dir", None) is None:
        args.data_dir = f"dataset/{args.split}"
    {"train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
