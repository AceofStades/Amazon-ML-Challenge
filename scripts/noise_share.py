"""Typo propagation test. Training, address-less true targets whose name core is shared by 2-10 entities: does the
target share a 'noise' token (in its name but not in the candidate entity's Source 1 name) with that entity's other
records more often for the owner than for other same-name entities?"""
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def toks(s):
    s = unicodedata.normalize("NFKD", s or "")
    return set(re.findall(r"[a-z0-9]+", "".join(ch for ch in s if not unicodedata.combining(ch)).lower()))


s1 = R("cache/train_source1.tsv", ["entity_id", "country", "name_core"]).with_columns(R("dataset/train/train_source1.tsv", ["business_name"])["business_name"].alias("sn"))
tg = pl.concat([R(f"cache/train_source{i}.tsv", ["entity_id", "country", "address", "name_core"]) for i in (2, 3)]).with_columns(
    pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"].alias("tn"))
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").select(pl.col("source1_entity_id").alias("e"), pl.col("matched_entity_ids").alias("t")))
k = s1.group_by("country", "name_core").agg(pl.col("entity_id").alias("cands"), pl.len().alias("k"))
na = tg.filter(pl.col("address") == "").join(gt, left_on="entity_id", right_on="t").join(k, on=["country", "name_core"])
na = na.filter((pl.col("k") >= 2) & (pl.col("k") <= 10)).select(pl.col("entity_id").alias("t"), pl.col("e").alias("owner"), "tn", "cands").sample(40000, seed=1)
pairs = na.explode("cands").rename({"cands": "e"}).with_columns((pl.col("e") == pl.col("owner")).alias("is_owner"))
pairs = pairs.join(s1.select(pl.col("entity_id").alias("e"), "sn"), on="e")
other = gt.join(tg.select(pl.col("entity_id").alias("t"), pl.col("tn").alias("on")), on="t")
agg = other.group_by("e").agg(pl.col("on"), pl.col("t").alias("ot"))
pairs = pairs.join(agg, on="e", how="left")


def shared(r):
    noise = toks(r["tn"]) - toks(r["sn"])
    if not noise:
        return -1
    for o, t in zip(r["on"] or [], r["ot"] or []):
        if t != r["t"] and noise & (toks(o) - toks(r["sn"])):
            return 1
    return 0


pairs = pairs.with_columns(pl.struct("tn", "sn", "on", "ot", "t").map_elements(shared, return_dtype=pl.Int8).alias("sh"))
print("targets with noise tokens (vs owner's S1 name):", pairs.filter(pl.col("is_owner") & (pl.col("sh") >= 0)).height, "of", na.height)
print(pairs.filter(pl.col("sh") >= 0).group_by("is_owner").agg(pl.len(), (pl.col("sh") == 1).mean().round(4).alias("shares_noise")).rows())
per = pairs.group_by("t").agg((pl.col("sh") == 1).sum().alias("n"), ((pl.col("sh") == 1) & pl.col("is_owner")).any().alias("own"))
u = per.filter(pl.col("n") == 1)
print("targets where exactly one candidate shares a noise token:", u.height, f"({u.height / per.height:.3f}), owner among them: {u['own'].mean():.3f}")
