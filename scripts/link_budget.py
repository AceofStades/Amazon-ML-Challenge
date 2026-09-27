"""Per-entity link budget by category, training truth against test submissions (US, India, France):
noaddr (target without address), then by street (address IDF cosine >= 0.6 or not), name similarity (name IDF cosine
>= 0.8 or not) and house relation (same / conflict / unknown). Shows where France accepts fewer links than US."""
import glob
import sys

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def categorize(split, pairs):
    """pairs: (si, ti) frame -> category column."""
    cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
    feats = pl.concat([pl.read_parquet(f, columns=cols).join(pairs, on=["si", "ti"], how="semi") for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet"))])
    addr = pl.concat([R(f"cache/{split}_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
    x = pairs.join(feats, on=["si", "ti"], how="left").with_columns(addr.gather(pairs["ti"].to_numpy()).alias("ta"))
    street = pl.when(pl.col("address_idf_cos") >= 0.6).then(pl.lit("street")).otherwise(pl.lit("otherst"))
    name = pl.when(pl.col("name_idf_cos") >= 0.8).then(pl.lit("simname")).otherwise(pl.lit("diffname"))
    house = pl.when(pl.col("house_match") == 1).then(pl.lit("same")).when(pl.col("house_conflict") == 1).then(pl.lit("conflict")).otherwise(pl.lit("unknown"))
    cat = (pl.when(pl.col("ta") == "").then(pl.lit("noaddr"))
           .when(pl.col("address_idf_cos").is_null()).then(pl.lit("notcand"))
           .otherwise(pl.concat_str([street, name, house], separator="/")))
    return x.with_columns(cat.alias("cat")).select("si", "ti", "cat")


def links(split, path):
    s1 = R(f"dataset/{split}/{split}_source1.tsv", ["entity_id", "country"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    l = (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
         .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
         .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti", "country"))
    E = s1.group_by("country").len().rename({"len": "E"})
    return l, E


out = []
l, E = links("train", "dataset/train/train_ground_truth.tsv")
c = categorize("train", l.select("si", "ti")).join(l, on=["si", "ti"])
out.append(c.group_by("country", "cat").len().join(E, on="country").with_columns(pl.concat_str([pl.lit("truth_"), pl.col("country")]).alias("col")))
l, E = links("test", sys.argv[1])
c = categorize("test", l.select("si", "ti")).join(l, on=["si", "ti"])
out.append(c.group_by("country", "cat").len().join(E, on="country").with_columns(pl.concat_str([pl.lit("sub_"), pl.col("country")]).alias("col")))
t = pl.concat(out).with_columns((pl.col("len") / pl.col("E")).round(4).alias("perE"))
w = t.pivot(on="col", index="cat", values="perE").fill_null(0).sort("cat")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(10); pl.Config.set_tbl_width_chars(200)
print(w)
print(w.select(pl.all().exclude("cat").sum()))
