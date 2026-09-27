"""Deep nearest neighbours for France Source 1 records against France targets (the stored kNN keeps the top 40,
which France's generic names saturate). Writes cache/fr_deep_knn.parquet (q = si, d = ti, rank, cosine) for
ranks 1..K. Usage: python scripts/fr_deep_knn.py [K]"""
import sys
import time

import numpy as np
import polars as pl
import torch

K = int(sys.argv[1]) if len(sys.argv) > 1 else 150
folder = "cache/embeddings"
s_ids = pl.read_parquet(f"{folder}/test_source1_ids.parquet").with_row_index("si")
country_col = "country" if "country" in s_ids.columns else None
R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/test/test_source1.tsv", ["country"]).with_row_index("si")
tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["country"]) for i in (2, 3)]).with_row_index("ti")
q_idx = s1.filter(pl.col("country") == "France")["si"].to_numpy()
t_idx = tg.filter(pl.col("country") == "France")["ti"].to_numpy()
print(f"France queries {len(q_idx):,}, France targets {len(t_idx):,}", flush=True)
s_vec = np.load(f"{folder}/test_source1.npy", mmap_mode="r")
t_vec = np.load(f"{folder}/test_targets.npy", mmap_mode="r")
dev = "cuda" if torch.cuda.is_available() else "cpu"
T = torch.from_numpy(np.ascontiguousarray(t_vec[t_idx])).to(dev).half()
started = time.time()
out_q, out_d, out_r, out_c = [], [], [], []
batch = 256
for b in range(0, len(q_idx), batch):
    qi = q_idx[b:b + batch]
    Q = torch.from_numpy(np.ascontiguousarray(s_vec[qi])).to(dev).half()
    sims = Q @ T.T
    val, pos = torch.topk(sims, K, dim=1)
    val = val.float()
    out_q.append(np.repeat(qi, K)); out_d.append(t_idx[pos.cpu().numpy()].ravel())
    out_r.append(np.tile(np.arange(1, K + 1, dtype=np.int16), len(qi))); out_c.append(val.cpu().numpy().ravel())
    if b % (batch * 200) == 0:
        print(f"  {b:,}/{len(q_idx):,} in {time.time() - started:.0f}s", flush=True)
df = pl.DataFrame({"q": np.concatenate(out_q).astype(np.uint32), "d": np.concatenate(out_d).astype(np.uint32),
                   "rank": np.concatenate(out_r), "cosine": np.concatenate(out_c).astype(np.float32)})
df.write_parquet("cache/fr_deep_knn.parquet")
old = pl.read_parquet(f"{folder}/test_knn.parquet").join(pl.DataFrame({"q": q_idx.astype(np.uint32)}), on="q", how="semi")
chk = df.filter(pl.col("rank") <= 40).join(old.select(pl.col("q").cast(pl.UInt32), pl.col("d").cast(pl.UInt32)), on=["q", "d"], how="semi").height
print(f"wrote {df.height:,} rows in {time.time() - started:.0f}s; overlap of our top 40 with stored kNN: {chk / max(1, df.filter(pl.col('rank') <= 40).height):.3f}")
