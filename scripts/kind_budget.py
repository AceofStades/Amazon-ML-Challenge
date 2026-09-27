"""Per-entity counts of name-change kinds (acronym, domain, alias, brand, exact) among training truth, all candidate
pairs, and accepted test links, by country. A kind far more frequent in France than in training, where the training
kind is nearly always true, is a candidate for a France-specific decoy process.
Usage: python scripts/kind_budget.py SUBMISSION"""
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)
KINDS = ["g_acronym", "g_domain", "g_alias", "g_brand1", "g_exact"]


def links(split, path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    col = "matched_entity_ids"
    l = (R(path).filter(pl.col(col) != "").with_columns(pl.col(col).str.split(",")).explode(col)
         .join(s1, left_on="source1_entity_id", right_on="entity_id").join(tg, left_on=col, right_on="entity_id").select("si", "ti"))
    return l.with_columns(pl.lit(1).alias("y")), s1.select("si", "country")


rows = []
for split, path in (("train", "dataset/train/train_ground_truth.tsv"), ("test", sys.argv[1])):
    l, s1 = links(split, path)
    g = pl.read_parquet(f"cache/gen/{split}.parquet", columns=["si", "ti"] + KINDS).join(s1, on="si", how="left")
    g = g.join(l, on=["si", "ti"], how="left").with_columns(pl.col("y").fill_null(0))
    E = s1.group_by("country").len().rename({"len": "E"})
    for k in KINDS:
        t = (g.filter(pl.col(k) > 0.5).group_by("country").agg(pl.len().alias("cand"), pl.col("y").sum().alias("linked"))
             .join(E, on="country").with_columns((pl.col("cand") / pl.col("E")).round(4).alias("cand_perE"),
                                                 (pl.col("linked") / pl.col("E")).round(4).alias("linked_perE"),
                                                 (pl.col("linked") / pl.col("cand")).round(3).alias("rate"))
             .with_columns(pl.lit(split).alias("split"), pl.lit(k).alias("kind")))
        rows.append(t.select("kind", "split", "country", "cand", "cand_perE", "linked", "linked_perE", "rate"))
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(200)
print(pl.concat(rows).sort("kind", "split", "country"))
