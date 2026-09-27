"""Class audit: same-street pairs with similar names, classified by house relation (same, gap0, near 1-30, mid 31-50,
far >50, unknown) and raw name difference (same/add/drop/adddrop after folding accents, case, dotted letters).
Training: per-entity count and true rate. Test (US, France): per-entity count and share accepted by a submission.
If France had training's per-entity true count in a class, its implied precision is train_true_perE / fr_perE."""
import glob
import re
import sys
import unicodedata

import polars as pl

R = lambda p, c=None: pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False, missing_utf8_is_empty_string=True, columns=c)


def fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).lower()


def tokens(s):
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
    return set(out)


def diff(sn, tn):
    a, b = tokens(sn), tokens(tn)
    return ("add" if b - a else "") + ("drop" if a - b else "") or "same"


def gap(h, ta):
    try:
        hv = int(h)
    except ValueError:
        return None
    nums = [int(n) for n in re.findall(r"\d+", ta) if len(n) <= 5]
    return min(abs(n - hv) for n in nums) if nums else None


def frame(split, country=None):
    cols = ["si", "ti", "house_match", "house_conflict", "address_idf_cos", "name_idf_cos"]
    s1 = R(f"cache/{split}_source1.tsv", ["country", "house"]).with_row_index("si")
    keep = s1.filter(pl.col("country") == country).select("si") if country else None
    parts = []
    for f in sorted(glob.glob(f"cache/v3/{split}/part*.parquet")):
        p = pl.read_parquet(f, columns=cols).filter((pl.col("address_idf_cos") >= 0.6) & (pl.col("name_idf_cos") >= 0.8))
        parts.append(p.join(keep, on="si", how="semi") if keep is not None else p)
    x = pl.concat(parts)
    addr = pl.concat([R(f"cache/{split}_source{i}.tsv", ["address"]) for i in (2, 3)])["address"].fill_null("")
    raw1 = R(f"dataset/{split}/{split}_source1.tsv", ["business_name"])["business_name"]
    raw2 = pl.concat([R(f"dataset/{split}/{split}_source{i}.tsv", ["business_name"]) for i in (2, 3)])["business_name"]
    x = x.with_columns(raw1.gather(x["si"].to_numpy()).alias("sn"), raw2.gather(x["ti"].to_numpy()).alias("tn"),
                       s1["house"].fill_null("").gather(x["si"].to_numpy()).alias("hs"), addr.gather(x["ti"].to_numpy()).alias("ta"))
    x = x.with_columns(pl.struct("sn", "tn").map_elements(lambda s: diff(s["sn"], s["tn"]), return_dtype=pl.String).alias("nd"),
                       pl.struct("hs", "ta").map_elements(lambda s: gap(s["hs"], s["ta"]), return_dtype=pl.Int64).alias("gap"))
    hrel = (pl.when(pl.col("house_match") == 1).then(pl.lit("same"))
            .when(pl.col("house_conflict") != 1).then(pl.lit("unknown"))
            .when(pl.col("gap") == 0).then(pl.lit("gap0"))
            .when(pl.col("gap") <= 30).then(pl.lit("near"))
            .when(pl.col("gap") <= 50).then(pl.lit("mid"))
            .otherwise(pl.lit("far")))
    return x.with_columns(hrel.alias("h")).select("si", "ti", "h", "nd")


def accepted(split_frame, path):
    s1 = R("dataset/test/test_source1.tsv", ["entity_id"]).with_row_index("si")
    tg = pl.concat([R(f"dataset/test/test_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
    links = (R(path).filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids").join(s1, left_on="source1_entity_id", right_on="entity_id")
             .join(tg, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("acc")))
    return split_frame.join(links, on=["si", "ti"], how="left").with_columns(pl.col("acc").fill_null(0))


sub = sys.argv[1]
tr = frame("train")
s1id = R("dataset/train/train_source1.tsv", ["entity_id"]).with_row_index("si")
tgid = pl.concat([R(f"dataset/train/train_source{i}.tsv", ["entity_id"]) for i in (2, 3)]).with_row_index("ti")
gt = (R("dataset/train/train_ground_truth.tsv").filter(pl.col("matched_entity_ids") != "").with_columns(pl.col("matched_entity_ids").str.split(","))
      .explode("matched_entity_ids").join(s1id, left_on="source1_entity_id", right_on="entity_id")
      .join(tgid, left_on="matched_entity_ids", right_on="entity_id").select("si", "ti").with_columns(pl.lit(1).alias("label")))
tr = tr.join(gt, on=["si", "ti"], how="left").with_columns(pl.col("label").fill_null(0))
T = tr.group_by("h", "nd").agg((pl.len() / 2206821).alias("tr_perE"), pl.col("label").mean().alias("tr_true"),
                               (pl.col("label").sum() / 2206821).alias("tr_true_perE"))
us = accepted(frame("test", "us"), sub).group_by("h", "nd").agg((pl.len() / 663106).alias("us_perE"), pl.col("acc").mean().alias("us_acc"))
fr = accepted(frame("test", "france"), sub)
fr.write_parquet("cache/class_audit_fr.parquet")
F = fr.group_by("h", "nd").agg((pl.len() / 259452).alias("fr_perE"), pl.col("acc").mean().alias("fr_acc"), pl.len().alias("fr_n"))
out = T.join(us, on=["h", "nd"], how="full", coalesce=True).join(F, on=["h", "nd"], how="full", coalesce=True)
out = out.with_columns((pl.col("tr_true_perE") / pl.col("fr_perE")).clip(0, 1).alias("fr_implied_prec"),
                       ((pl.col("tr_perE") - pl.col("tr_true_perE")) * 2 / pl.col("us_perE")).alias("us_decoy_x2_share"))
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(14); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(4)
print(out.sort("h", "nd").select("h", "nd", "tr_perE", "tr_true", "us_perE", "us_acc", "fr_n", "fr_perE", "fr_acc", "fr_implied_prec"))
