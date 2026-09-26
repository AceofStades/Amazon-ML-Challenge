"""Kaggle GPU job: score (si, ti) record pairs with the fine-tuned cross-encoder.

Inputs: the model saved by the amzn-ce-train kernel (ce_model/), record texts
(train_*/test_* parquet) and pair lists pairs_{split}.parquet (si, ti).
One process per GPU scores an interleaved share of length-sorted batches.
Output: scores_{split}.parquet with si, ti, ce_logit.
"""

import glob
import os
import time

import numpy as np
import polars as pl
import torch
import torch.multiprocessing as mp
from transformers import AutoModel, AutoTokenizer

MAX_LENGTH = 128
BATCH = 256
SPLITS = ("test", "train")


def find(name: str) -> str:
    matches = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


def mean_pool(output, mask):
    mask = mask.unsqueeze(-1).to(output.dtype)
    return (output * mask).sum(1) / mask.sum(1).clamp(min=1)


def worker(rank: int, world: int, model_dir: str, split: str, out: str):
    torch.cuda.set_device(rank)
    pairs = pl.read_parquet(find(f"pairs_{split}.parquet"), columns=["si", "ti"])
    s_text = pl.read_parquet(find(f"{split}_s1.parquet"), columns=["text"])["text"]
    t_text = pl.read_parquet(find(f"{split}_t.parquet"), columns=["text"])["text"]
    lengths = (s_text.str.len_chars().gather(pairs["si"].to_numpy()).to_numpy()
               + t_text.str.len_chars().gather(pairs["ti"].to_numpy()).to_numpy())
    order = np.argsort(lengths, kind="stable")
    batches = [order[s:s + BATCH] for s in range(0, len(order), BATCH)][rank::world]
    mine = np.concatenate(batches) if batches else np.zeros(0, np.int64)
    a = s_text.gather(pairs["si"].to_numpy()[mine]).to_list()
    b = t_text.gather(pairs["ti"].to_numpy()[mine]).to_list()
    del s_text, t_text, pairs
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16, add_pooling_layer=False).cuda(rank).eval()
    head = torch.nn.Linear(model.config.hidden_size, 1)
    head.load_state_dict(torch.load(os.path.join(model_dir, "head.pt"), map_location="cpu"))
    head = head.half().cuda(rank)
    scores = np.empty(len(mine), np.float32)
    started = time.time()
    with torch.inference_mode():
        for n, start in enumerate(range(0, len(mine), BATCH)):
            batch = tokenizer(a[start:start + BATCH], b[start:start + BATCH], padding=True, truncation=True,
                              max_length=MAX_LENGTH, return_tensors="pt").to(rank)
            hidden = model(**batch).last_hidden_state
            logits = head(mean_pool(hidden, batch["attention_mask"])).squeeze(-1)
            scores[start:start + BATCH] = logits.float().cpu().numpy()
            if rank == 0 and n % 1000 == 0:
                done = (start + BATCH) * world
                print(f"{split}: ~{done:,}/{len(order):,} pairs, {done / (time.time() - started):.0f}/s", flush=True)
    np.save(f"{out}.rank{rank}.index.npy", mine)
    np.save(f"{out}.rank{rank}.score.npy", scores)


if __name__ == "__main__":
    model_dir = os.path.dirname(find("ce_model/head.pt"))
    print("model", model_dir, "GPUs", torch.cuda.device_count(), flush=True)
    world = max(1, torch.cuda.device_count())
    for split in SPLITS:
        try:
            pairs = pl.read_parquet(find(f"pairs_{split}.parquet"), columns=["si", "ti"])
        except FileNotFoundError:
            print("no pairs for", split, flush=True)
            continue
        out = f"/kaggle/working/scores_{split}"
        mp.spawn(worker, args=(world, model_dir, split, out), nprocs=world, join=True)
        index = np.concatenate([np.load(f"{out}.rank{r}.index.npy") for r in range(world)])
        score = np.concatenate([np.load(f"{out}.rank{r}.score.npy") for r in range(world)])
        result = np.full(pairs.height, np.nan, np.float32)
        result[index] = score
        pairs.with_columns(ce_logit=result).write_parquet(f"{out}.parquet")
        for r in range(world):
            os.remove(f"{out}.rank{r}.index.npy")
            os.remove(f"{out}.rank{r}.score.npy")
        print(f"wrote {out}.parquet: {pairs.height:,} pairs", flush=True)
