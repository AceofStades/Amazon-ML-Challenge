"""Training true links missing from the candidate set (5% entity sample): what they look like (address-less, same
normalized address, name overlap), to judge whether an extra candidate source could recover them."""
import glob

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
s1 = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
      .join(tid, left_on="matched_entity_ids", right_on="entity_id").select(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32)))
ent = pl.concat([pl.read_parquet(f, columns=["si"]).unique() for f in sorted(glob.glob("cache/v3/train/part*.parquet"))]).unique()
gt = gt.join(ent.with_columns(pl.col("si").cast(pl.UInt32)), on="si", how="semi").filter(pl.col("si").hash(5) % 20 == 0)
cand = pl.concat([pl.read_parquet(f, columns=["si", "ti"]).join(gt.select("si").unique(), on="si", how="semi") for f in sorted(glob.glob("cache/v3/train/part*.parquet"))])
miss = gt.join(cand.with_columns(pl.col("si").cast(pl.UInt32), pl.col("ti").cast(pl.UInt32)), on=["si", "ti"], how="anti")
print(f"true links {gt.height:,}, missing from candidates {miss.height:,} ({miss.height / gt.height:.4f})")
S = R("cache/train_source1.tsv", ["country", "name_core", "address_core", "house"]).with_row_index("si").with_columns(pl.col("si").cast(pl.UInt32))
T = pl.concat([R(f"cache/train_source{i}.tsv", ["name_core", "address_core", "address"]) for i in (2, 3)]).with_row_index("ti").with_columns(pl.col("ti").cast(pl.UInt32))
raw1 = R("dataset/train/train_source1.tsv", ["business_name", "business_address"]).with_row_index("si").with_columns(pl.col("si").cast(pl.UInt32))
raw2 = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name", "business_address"]) for i in (2, 3)]).with_row_index("ti").with_columns(pl.col("ti").cast(pl.UInt32))
m = miss.join(S, on="si").join(T.rename({"name_core": "t_core", "address_core": "t_addr"}), on="ti")
m = m.with_columns((pl.col("address") == "").alias("noaddr"), (pl.col("address_core") == pl.col("t_addr")).alias("addr_eq"),
                   (pl.col("name_core") == pl.col("t_core")).alias("core_eq"))
print(m.group_by("country", "noaddr", "addr_eq", "core_eq").len().sort("len", descending=True).head(12).rows())
ex = m.sample(min(16, m.height), seed=2).join(raw1.rename({"business_name": "sn", "business_address": "sa"}), on="si").join(
    raw2.rename({"business_name": "tn", "business_address": "ta"}), on="ti")
for r in ex.iter_rows(named=True):
    print(f"  {r['sn'][:28]:28} {r['sa'][:30]:30} || {r['tn'][:28]:28} {r['ta'][:30]}")
