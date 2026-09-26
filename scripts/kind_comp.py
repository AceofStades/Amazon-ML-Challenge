"""Per-entity counts of true records by noise kind (domain, acronym, alias, single brand word) in training."""
import re
import sys

import polars as pl

sys.path.insert(0, "src")
from amzn.france_rule import kind  # noqa: E402

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
g = (R("dataset/train/train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
     .filter(pl.col("matched_entity_ids") != ""))
t = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id", "business_name"]) for i in (2, 3)])
sample = g.select("source1_entity_id").unique().sample(300000, seed=1)
g = g.join(sample, on="source1_entity_id", how="semi").join(t, left_on="matched_entity_ids", right_on="entity_id")


def k(name):
    base = kind(name)
    if base != "other":
        return base
    words = re.findall(r"[A-Za-z]+", name or "")
    return "brand" if len(words) == 1 and len(words[0]) >= 6 else "other"


g = g.with_columns(pl.col("business_name").map_elements(k, return_dtype=pl.String).alias("kind"))
per = g.group_by("source1_entity_id", "kind").len()
for kd in ("domain", "acronym", "alias", "brand"):
    c = per.filter(pl.col("kind") == kd)["len"]
    tot = sample.height
    print(f"{kd}: entities with 1: {(c == 1).sum() / tot:.4f}, with 2+: {(c >= 2).sum() / tot:.4f}")
