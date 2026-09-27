"""Sanity of every cross-encoder score file: rows, score column, and rank correlation with the reference large
cross-encoder (cache/ce_large) on shared pairs, on train and on test (US/India vs France separately). A model whose
test correlation collapses while train stays high has misaligned or broken test scores."""
import glob

import polars as pl

ids = pl.read_parquet("cache/embeddings/test_source1_ids.parquet").select("country").with_row_index("si")


def load(d, split):
    x = pl.read_parquet(f"{d}/{split}_scores.parquet")
    col = [c for c in x.columns if c not in ("si", "ti")][0]
    return x.select("si", "ti", pl.col(col).cast(pl.Float64).alias("s"))


ref = {s: load("cache/ce_large", s) for s in ("train", "test")}
rows = []
for d in sorted(glob.glob("cache/ce_*")):
    if not glob.glob(f"{d}/test_scores.parquet") or not glob.glob(f"{d}/train_scores.parquet"):
        continue
    out = {"ce": d.split("ce_", 1)[1]}
    for s in ("train", "test"):
        x = load(d, s).join(ref[s].rename({"s": "r"}), on=["si", "ti"])
        out[f"{s}_n"] = x.height
        if s == "test":
            x = x.join(ids, on="si")
            for c, sub in (("usin", x.filter(pl.col("country") != "France")), ("fr", x.filter(pl.col("country") == "France"))):
                out[f"test_{c}_rho"] = round(sub.select(pl.corr("s", "r", method="spearman")).item(), 3)
                out[f"test_{c}_mean"] = round(sub["s"].mean(), 3)
        else:
            out["train_rho"] = round(x.select(pl.corr("s", "r", method="spearman")).item(), 3)
            out["train_mean"] = round(x["s"].mean(), 3)
    rows.append(out)
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(12); pl.Config.set_tbl_width_chars(200)
print(pl.DataFrame(rows))
