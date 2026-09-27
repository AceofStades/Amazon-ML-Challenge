"""Can the owner of an address-less target be told apart from other Source 1 entities with the same name core?
Training, address-less true targets whose name core is shared by 2-10 entities: for each candidate entity, raw-name
agreement signals with the entity's Source 1 record and with its other (addressed) target records."""
import re
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokset(s):
    out, run = [], ""
    for w in re.findall(r"[a-z0-9]+", fold(s)):
        if len(w) == 1 and w.isalpha():
            run += w
            continue
        if run:
            out.append(run); run = ""
        out.append(w)
    if run:
        out.append(run)
    return " ".join(sorted(set(out)))


s1 = R("cache/train_source1.tsv", ["entity_id", "country", "name_core"]).with_columns(
    R("dataset/train/train_source1.tsv", ["business_name"])["business_name"].alias("raw"))
tg = pl.concat([R(f"cache/train_source{i}.tsv", ["entity_id", "country", "address", "name_core"]) for i in (2, 3)]).with_columns(
    pl.concat([R(f"dataset/train/train_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"].alias("raw"))
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").select(pl.col("source1_entity_id").alias("e"), pl.col("matched_entity_ids").alias("t")))
k = s1.group_by("country", "name_core").agg(pl.col("entity_id").alias("cands"), pl.len().alias("k"))
na = tg.filter(pl.col("address") == "").join(gt, left_on="entity_id", right_on="t").join(k, on=["country", "name_core"])
na = na.filter((pl.col("k") >= 2) & (pl.col("k") <= 10)).select(pl.col("entity_id").alias("t"), pl.col("e").alias("owner"), "raw", "k", "cands")
print("address-less true targets with 2-10 same-core entities:", na.height)
pairs = na.explode("cands").rename({"cands": "e"}).with_columns((pl.col("e") == pl.col("owner")).alias("is_owner"))
pairs = pairs.join(s1.select(pl.col("entity_id").alias("e"), pl.col("raw").alias("s_raw")), on="e")
# other records of each candidate entity (true links in training; in test these would be the confident matches)
other = gt.join(tg.select(pl.col("entity_id").alias("t"), pl.col("raw").alias("o_raw")), on="t")
other = other.with_columns(pl.col("o_raw").map_elements(fold, return_dtype=pl.String).alias("o_fold"),
                           pl.col("o_raw").map_elements(tokset, return_dtype=pl.String).alias("o_tok"))
pairs = pairs.with_columns(pl.col("raw").map_elements(fold, return_dtype=pl.String).alias("t_fold"),
                           pl.col("raw").map_elements(tokset, return_dtype=pl.String).alias("t_tok"),
                           pl.col("s_raw").map_elements(fold, return_dtype=pl.String).alias("s_fold"),
                           pl.col("s_raw").map_elements(tokset, return_dtype=pl.String).alias("s_tok"))
agg = other.group_by("e").agg(pl.col("o_raw"), pl.col("o_fold"), pl.col("o_tok"), pl.col("t").alias("o_t"))
pairs = pairs.join(agg, on="e", how="left")
pairs = pairs.with_columns(
    (pl.col("raw") == pl.col("s_raw")).alias("s_exact"),
    (pl.col("t_fold") == pl.col("s_fold")).alias("s_fold_eq"),
    (pl.col("t_tok") == pl.col("s_tok")).alias("s_tok_eq"),
    pl.struct("raw", "o_raw", "t", "o_t").map_elements(lambda r: any(o == r["raw"] for o, ot in zip(r["o_raw"] or [], r["o_t"] or []) if ot != r["t"]),
                                                     return_dtype=pl.Boolean).alias("o_exact"),
    pl.struct("t_tok", "o_tok", "t", "o_t").map_elements(lambda r: any(o == r["t_tok"] for o, ot in zip(r["o_tok"] or [], r["o_t"] or []) if ot != r["t"]),
                                                       return_dtype=pl.Boolean).alias("o_tok_eq"))
for sig in ("s_exact", "s_fold_eq", "s_tok_eq", "o_exact", "o_tok_eq"):
    t = pairs.group_by("is_owner").agg(pl.col(sig).mean().round(4).alias("rate"))
    per_t = pairs.group_by("t").agg(pl.col(sig).sum().alias("n_sig"), (pl.col(sig) & pl.col("is_owner")).any().alias("owner_sig"))
    uniq = per_t.filter(pl.col("n_sig") == 1)
    print(f"{sig}: owner rate {t.filter(pl.col('is_owner'))['rate'][0]}, non-owner rate {t.filter(~pl.col('is_owner'))['rate'][0]}; "
          f"targets with exactly one candidate showing it: {uniq.height} ({uniq.height / per_t.height:.3f}), owner among them {uniq['owner_sig'].mean():.3f}")
