"""Address-less targets are almost always owned by some Source 1 entity, so the owner posterior should be renormalized
over the target's candidates: q = top1 / sum(p1). Training (full competition, out-of-fold p1): precision of the top
candidate by q and top1 cells. Test: how many currently unassigned address-less targets fall in each cell, by country."""
import glob
import sys

import polars as pl

R = lambda p, c: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
QB = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
TB = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def target_table(split):
    tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["entity_id", "address", "country"]) for i in (2, 3)]).with_row_index("ti")
    noaddr = tg.filter(pl.col("address") == "").select("ti", pl.col("entity_id").alias("t_id"), "country")
    parts = [pl.read_parquet(f).join(noaddr.select("ti"), on="ti", how="semi") for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))]
    return noaddr, pl.concat(parts)


def summarize(p):
    p = p.with_columns(pl.col("p1").rank("ordinal", descending=True).over("ti").alias("r"))
    agg = [pl.col("p1").max().alias("top1"), pl.col("p1").sum().alias("S"), pl.col("si").filter(pl.col("r") == 1).first().alias("top_si")]
    if "label" in p.columns:
        agg.append((pl.col("label") * (pl.col("r") == 1)).max().alias("top_is_owner"))
    t = p.group_by("ti").agg(agg).with_columns((pl.col("top1") / pl.col("S")).alias("q"))
    return t.with_columns(pl.col("q").cut(QB).alias("qb"), pl.col("top1").cut(TB).alias("tb"))


# Training with labels (cached after the first run).
import os
if os.path.exists("cache/noaddr_q_train.parquet"):
    tr = pl.read_parquet("cache/noaddr_q_train.parquet")
else:
    tr = None
noaddr, p = target_table("train") if tr is None else (None, None)
if tr is None:
  s1 = R("cache/train_source1.tsv", ["entity_id"]).with_row_index("si")
  gt = (R("dataset/train/train_ground_truth.tsv", None).filter(pl.col("matched_entity_ids") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
        .select(pl.col("source1_entity_id").alias("s_id"), pl.col("matched_entity_ids").alias("t_id")))
  gt = gt.join(noaddr, on="t_id").join(s1.rename({"entity_id": "s_id"}), on="s_id").select("si", "ti").with_columns(pl.lit(1).alias("label"))
  p = p.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
  tr = summarize(p)
  tr.write_parquet("cache/noaddr_q_train.parquet")
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(12)
cell = tr.group_by("tb", "qb").agg(pl.len().alias("n_train"), pl.col("top_is_owner").mean().round(3).alias("prec"))

# Test: unassigned address-less targets in the given submission.
noaddr_t, pt = target_table("test")
tt = summarize(pt).join(noaddr_t.select("ti", "country"), on="ti")
tgt = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
sub = (R(sys.argv[1], None).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
       .explode("matched_entity_ids").select(pl.col("matched_entity_ids").alias("entity_id")))
assigned = tgt.join(sub, on="entity_id", how="semi").select("ti")
tt = tt.join(assigned, on="ti", how="anti")
tt.write_parquet("cache/noaddr_q_test_unassigned.parquet")
cnt = tt.group_by("tb", "qb").agg([(pl.col("country") == c).sum().alias(c) for c in ("us", "india", "france")])
out = cell.join(cnt, on=["tb", "qb"], how="full", coalesce=True).sort("tb", "qb")
print(out.filter(pl.col("n_train").fill_null(0) + pl.col("us").fill_null(0) > 50))
