"""Is an acronym record one of the entity's usual records, or an extra one? Compares, per country, the number of links
of entities with a same-house consistent acronym link against entities without one (training truth for US/India,
accepted links for test). True noise replaces one record (with-acronym entities have only slightly more links, from
size bias); an extra decoy adds one link on top.
Usage: python scripts/acronym_extra.py SUBMISSION"""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
KINDS = ["g_acronym", "g_consistent", "g_brand1", "g_domain", "g_alias", "g_exact"]


def run(split, path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1.select("entity_id", "si"), left_on="source1_entity_id", right_on="entity_id")
         .join(tg.select("entity_id", "ti"), left_on=col, right_on="entity_id").select("si", "ti"))
    g = pl.read_parquet(f"cache/gen/{split}.parquet", columns=["si", "ti"] + KINDS)
    lg = l.join(g, on=["si", "ti"], how="inner")   # links that were scored (have gen features)
    ents = lg.select("si").unique()                 # entities covered by the gen cache
    n = l.join(ents, on="si", how="semi").group_by("si").len().rename({"len": "n"})
    flags = lg.group_by("si").agg(((pl.col("g_acronym") > 0.5) & (pl.col("g_consistent") > 0.5)).any().alias("acro"),
                                  (pl.col("g_brand1") > 0.5).any().alias("brand"), (pl.col("g_domain") > 0.5).any().alias("domain"),
                                  (pl.col("g_alias") > 0.5).any().alias("alias"))
    x = n.join(flags, on="si").join(s1.select("si", "country"), on="si")
    out = []
    for k in ("acro", "brand", "domain", "alias"):
        t = x.group_by("country", k).agg(pl.len().alias("ents"), pl.col("n").mean().round(3).alias("mean_links")).sort("country", k)
        t = t.pivot(on=k, index="country", values=["ents", "mean_links"]).with_columns(pl.lit(k).alias("kind"), pl.lit(split).alias("split"))
        out.append(t)
    return pl.concat(out, how="diagonal")


pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(200)
print(pl.concat([run("train", "dataset/train/train_ground_truth.tsv"), run("test", sys.argv[1])], how="diagonal")
      .select("split", "kind", "country", "ents_false", "ents_true", "mean_links_false", "mean_links_true")
      .with_columns((pl.col("mean_links_true") - pl.col("mean_links_false")).round(3).alias("diff")).sort("kind", "split", "country"))
