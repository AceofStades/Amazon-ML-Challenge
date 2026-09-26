"""Kaggle GPU job: fine-tune a multilingual cross-encoder for business record pairs.

Inputs (private dataset): train_s1.parquet / train_t.parquet (record texts in
index order) and ce_train_pairs.parquet (si, ti, label). The base model is an
MIT-licensed multilingual encoder; a linear head on the mean-pooled output
gives the match logit. Runs DDP across all visible GPUs with fp16 autocast and
stops at a wall-clock budget, always saving the model.
"""

import glob
import math
import os
import time

import numpy as np
import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def find(name: str) -> str:
    matches = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def base_model() -> str:
    """Base weights shipped in the attached dataset (kernels without internet), else the hub id."""
    shipped = glob.glob("/kaggle/input/**/model.safetensors", recursive=True)
    return os.path.dirname(shipped[0]) if shipped else "intfloat/multilingual-e5-large"


CONFIG = {
    "model": base_model(),
    "max_length": 128,
    "batch_per_gpu": 40,
    "lr": 2e-5,
    "max_pairs": 3_000_000,
    "valid_pairs": 30_000,
    "budget_hours": 8.5,
    "output": "/kaggle/working/ce_model",
}
STARTED = time.time()




def mean_pool(output: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(output.dtype)
    return (output * mask).sum(1) / mask.sum(1).clamp(min=1)


class CrossEncoder(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(name, add_pooling_layer=False)
        self.head = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, batch):
        output = self.model(**batch).last_hidden_state
        return self.head(mean_pool(output, batch["attention_mask"])).squeeze(-1)


def load_pairs():
    pairs = pl.read_parquet(find("ce_train_pairs_hard_seed41.parquet"))
    pairs = pairs.head(CONFIG["max_pairs"] + CONFIG["valid_pairs"])
    s_text = pl.read_parquet(find("train_s1.parquet"), columns=["text"])["text"]
    t_text = pl.read_parquet(find("train_t.parquet"), columns=["text"])["text"]
    a = s_text.gather(pairs["si"].to_numpy()).to_list()
    b = t_text.gather(pairs["ti"].to_numpy()).to_list()
    return a, b, pairs["label"].to_numpy().astype(np.float32)


def batches(lengths: np.ndarray, batch: int, rank: int, world: int, seed: int):
    """Length-bucketed batches: sort within chunks of 50 batches, shuffle batch order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lengths))
    chunk = batch * world * 50
    result = []
    for start in range(0, len(order), chunk):
        block = order[start:start + chunk]
        block = block[np.argsort(lengths[block], kind="stable")]
        for b in range(0, len(block) - batch * world + 1, batch * world):
            result.append(block[b:b + batch * world])
    rng.shuffle(result)
    return [r[rank::world] for r in result]


def worker(rank: int, world: int, a, b, labels) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"])
    model = CrossEncoder(CONFIG["model"])
    # The 250k-token embedding matrix is half of the parameters; freezing it keeps
    # optimizer state within a 15 GB T4.
    model.model.embeddings.word_embeddings.weight.requires_grad_(False)
    model = model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    valid = CONFIG["valid_pairs"]
    train_index = np.arange(valid, len(labels))
    lengths = np.fromiter((len(a[i]) + len(b[i]) for i in train_index), np.int32, len(train_index))
    plan = batches(lengths, CONFIG["batch_per_gpu"], rank, world, seed=1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG["lr"],
                                  weight_decay=0.01, foreach=False)
    total = len(plan)
    warmup = max(1, total // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    running = None
    for step, rows in enumerate(plan):
        rows = train_index[rows]
        batch = tokenizer([a[i] for i in rows], [b[i] for i in rows], padding=True, truncation=True,
                          max_length=CONFIG["max_length"], return_tensors="pt").to(rank)
        target = torch.from_numpy(labels[rows]).cuda(rank)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits.float(), target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        elapsed = time.time() - STARTED
        stop = torch.tensor([elapsed > CONFIG["budget_hours"] * 3600], device=rank, dtype=torch.int)
        dist.all_reduce(stop)
        if rank == 0 and step % 200 == 0:
            rate = (step + 1) * CONFIG["batch_per_gpu"] * world / max(elapsed, 1)
            print(f"step {step}/{total} loss {running:.4f} pairs/s {rate:.0f} elapsed {elapsed / 3600:.2f}h", flush=True)
        if stop.item():
            if rank == 0:
                print(f"time budget reached at step {step}/{total}", flush=True)
            break
    if rank == 0:
        model.eval()
        inner = model.module
        scores = []
        with torch.inference_mode():
            for start in range(0, valid, 256):
                rows = np.arange(start, min(start + 256, valid))
                batch = tokenizer([a[i] for i in rows], [b[i] for i in rows], padding=True, truncation=True,
                                  max_length=CONFIG["max_length"], return_tensors="pt").to(rank)
                with torch.autocast("cuda", dtype=torch.float16):
                    scores.append(inner(batch).float().cpu().numpy())
        scores = np.concatenate(scores)
        y = labels[:valid]
        p = 1 / (1 + np.exp(-scores))
        logloss = -np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))
        accuracy = np.mean((p > 0.5) == (y > 0.5))
        print(f"validation logloss {logloss:.5f} accuracy {accuracy:.5f} on {valid} pairs", flush=True)
        os.makedirs(CONFIG["output"], exist_ok=True)
        inner.model.half().save_pretrained(CONFIG["output"])
        tokenizer.save_pretrained(CONFIG["output"])
        torch.save(inner.head.state_dict(), os.path.join(CONFIG["output"], "head.pt"))
        with open(os.path.join(CONFIG["output"], "config_ce.txt"), "w") as handle:
            handle.write(repr(CONFIG) + f"\nlogloss {logloss}\naccuracy {accuracy}\n")
        print("saved", CONFIG["output"], flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print("base model:", CONFIG["model"], flush=True)
    print(torch.cuda.device_count(), "GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
    AutoTokenizer.from_pretrained(CONFIG["model"])
    AutoModel.from_pretrained(CONFIG["model"])  # download once before workers start
    a, b, labels = load_pairs()
    print(f"pairs {len(labels):,} positive rate {labels.mean():.3f}", flush=True)
    world = max(1, torch.cuda.device_count())
    mp.spawn(worker, args=(world, a, b, labels), nprocs=world, join=True)
