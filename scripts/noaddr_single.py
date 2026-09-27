"""Address-less targets with exactly one candidate entity (stage-1 p1 >= 0.01): training precision of assigning the
target to it, by p1 band; France test count of such targets left unassigned by the submission."""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
bands = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


def single(split):
    tg = pl.concat([R(f"cache/{split}_source{i}.tsv", ["address"]) for i in (2, 3)]).with_row_index("ti").filter(pl.col("address") == "").select("ti")
    p = pl.concat([pl.read_parquet(f).join(tg, on="ti", how="semi").filter(pl.col("p1") >= 0.01) for f in sorted(glob.glob(f"output/v7/{split}_p1/*.parquet"))])
    return p.filter(pl.len().over("ti") == 1)


tr = single("train")
s1 = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
      .join(tid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
tr = tr.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0), pl.col("p1").cut(bands).alias("b"))
print("train single-candidate address-less targets:", tr.height)
print(tr.group_by("b").agg(pl.len(), pl.col("label").mean().round(4)).sort("b").rows())
te = single("test")
s1t = R("dataset/test/test_source1.tsv", ["entity_id", "country"]).with_row_index("si")
tgt = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
links = (R(sys.argv[1]).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1t.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tgt, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti"))
te = te.join(links.select("ti").unique(), on="ti", how="anti").join(s1t.select("si", "country"), on="si").with_columns(pl.col("p1").cut(bands).alias("b"))
print("test unassigned single-candidate address-less targets by country and band:")
print(te.group_by("country", "b").len().pivot(on="country", index="b", values="len").fill_null(0).sort("b"))
te.write_parquet("cache/noaddr_single_test.parquet")
